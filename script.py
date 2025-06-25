#!/usr/bin/env python3
"""
Whisper transcription CLI

Features
--------
• API mode (default)   – uses OpenAI Whisper
• Local mode           – `--mode local --local-model medium` loads openai-whisper
• Automatic device     – CUDA → MPS → CPU, with Sparse-MPS → CPU fallback
• Optional diarization – `--diarize` (needs HuggingFace token & licence accept)
• --timestamps / --aggregate flags
• Robust FFmpeg progress, chunking, retries
"""

from __future__ import annotations
import argparse, itertools, os, re, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import httpx, torch
from pydub import AudioSegment
from tqdm import tqdm
import inspect

# ------------------------------ ENV -----------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# ------------------------------ CONSTANTS -----------------------------------
MAX_CONC_REQUESTS, PER_REQ_TIMEOUT = 4, 90
MAX_MB, SAFETY, BITRATE = 25, 0.95, 128
TARGET_BYTES = int(MAX_MB * 1_048_576 * SAFETY)
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE / 8 * 1024))
WHISPER_CACHE = os.path.join(os.path.dirname(__file__), "models", "whisper")


# ------------------------------ HELPERS -------------------------------------
def log(msg: str):
    print(msg, flush=True)


def prefer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Spinner:
    def __init__(self, desc):
        self.desc = desc
        self.stop = threading.Event()

    def __enter__(self):
        log(self.desc)
        threading.Thread(target=self._spin, daemon=True).start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        sys.stdout.write("\r" + " " * (len(self.desc) + 4) + "\r")
        sys.stdout.flush()

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self.stop.is_set():
                break
            sys.stdout.write(f"\r{self.desc} {ch}")
            sys.stdout.flush()
            time.sleep(0.1)


# ------------------------------ FFMPEG --------------------------------------
T_RE = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")


def _dur(p):
    try:
        return float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    p,
                ],
                text=True,
            ).strip()
        )
    except Exception:
        return None


def _run(cmd: list[str], total: Optional[float], desc: str):
    if not total:
        with tqdm(
            total=1, desc=desc, bar_format="{l_bar}{bar}|{elapsed}", ncols=80
        ) as pb:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
            pb.update(1)
            return
    with tqdm(total=int(total), desc=desc, unit="s", ncols=80) as pb:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            if m := T_RE.search(line):
                h, mn, s = m.groups()
                pb.n = min(int(float(s) + int(mn) * 60 + int(h) * 3600), pb.total)
                pb.refresh()
        proc.wait()
        pb.n = pb.total
        pb.refresh()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def patch_torchaudio():
    """Patch ``torchaudio.load`` for compatibility with pyannote >= 3.1."""

    try:
        import torchaudio
    except Exception:
        return

    sig = inspect.signature(torchaudio.load)
    if "backend" in sig.parameters:
        return

    original = torchaudio.load

    def load(path, *args, backend=None, **kwargs):
        if backend == "soundfile":
            import soundfile as sf

            data, sr = sf.read(path, always_2d=True, dtype="float32")
            return torch.from_numpy(data.T), sr
        return original(path, *args, **kwargs)

    torchaudio.load = load


def convert(src: str) -> str:
    """Return ``src`` if already MP3, otherwise convert and return new file."""

    if src.lower().endswith(".mp3"):
        return src

    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    _run(["ffmpeg", "-y", "-i", src, dst], _dur(src), "[STEP] Converting → MP3")
    return dst


def remove_silence(p):
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    filt = (
        "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:"
        "stop_periods=-1:stop_duration=0.3:stop_threshold=-50dB"
    )
    _run(
        ["ffmpeg", "-y", "-i", p, "-af", filt, dst], _dur(p), "[STEP] Removing silence"
    )
    audio = AudioSegment.from_file(dst)
    os.unlink(dst)
    return audio


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _estimate_chunks(audio: AudioSegment) -> int:
    """Estimate number of chunks produced by :func:`chunk`."""

    lim = SECONDS_LIMIT * 1000
    total = len(audio)
    start = 0
    count = 0
    while start < total:
        end = min(start + lim, total)
        if end < total and audio[end - 2000 : end].dBFS < -45:
            end -= 2000
        start = end
        count += 1
    return count


def chunk(audio: AudioSegment):
    lim = SECONDS_LIMIT * 1000
    total = len(audio)
    start = 0
    chunks = []
    with tqdm(
        total=_estimate_chunks(audio), desc="[STEP] Chunking", unit="chunk", ncols=80
    ) as pb:
        while start < total:
            end = min(start + lim, total)
            if end < total and audio[end - 2000 : end].dBFS < -45:
                end -= 2000
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            audio[start:end].export(tmp, format="mp3", bitrate=f"{BITRATE}k")
            chunks.append((tmp, start / 1000, end / 1000))
            start = end
            pb.update(1)
    return chunks


# ------------------------------ LOCAL WHISPER -------------------------------
try:
    import whisper
except ImportError:
    whisper = None
_MODEL_CACHE = {}


def load_local(tag: str, dev: torch.device):
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]
    if whisper is None:
        raise RuntimeError(
            "Local mode requested but `openai-whisper` is not installed."
        )
    try:
        log(f"[STEP] Loading Whisper '{tag}' on {dev} …")
        mdl = whisper.load_model(tag, device=str(dev), download_root=WHISPER_CACHE)
    except (RuntimeError, NotImplementedError) as e:
        if "SparseMPS" in str(e) or "_sparse_coo_tensor" in str(e):
            log("[WARN] Sparse op missing on MPS – falling back to CPU")
            mdl = whisper.load_model(tag, device="cpu", download_root=WHISPER_CACHE)
        else:
            raise
    _MODEL_CACHE[tag] = mdl
    return mdl


def w_tx_local(m, p, lang):
    return m.transcribe(p, language=lang, fp16=(m.device.type != "cpu"))["text"].strip()


# ------------------------------ OPENAI API ----------------------------------
try:
    import openai
except ImportError:
    openai = None


def w_tx_api(cli, p, lang, tries=5):
    for i in range(1, tries + 1):
        try:
            with open(p, "rb") as f:
                return cli.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                    timeout=PER_REQ_TIMEOUT,
                    language=lang,
                ).strip()
        except (openai.APITimeoutError, openai.RateLimitError) as e:
            back = 2**i
            log(f"[WARN] {e} – retry {i}/{tries} in {back}s")
            time.sleep(back)
    raise RuntimeError("Whisper API failed repeatedly")


# ------------------------------ CHUNK TRANSCR. ------------------------------
def transcribe_chunks(chs, *, mode, cli=None, mdl=None, lang):
    res = [None] * len(chs)

    def work(idx_pair):
        i, (p, st, ed) = idx_pair
        try:
            txt = w_tx_api(cli, p, lang) if mode == "api" else w_tx_local(mdl, p, lang)
        finally:
            os.unlink(p)
        return i, None, txt, st, ed

    workers = MAX_CONC_REQUESTS if mode == "api" else 1
    workers = min(workers, len(chs)) or 1
    with tqdm(
        total=len(chs), desc="[STEP] Transcribing", unit="chunk", ncols=80
    ) as pb, ThreadPoolExecutor(workers) as pool:
        for fut in as_completed(pool.submit(work, x) for x in enumerate(chs)):
            i, *row = fut.result()
            res[i] = tuple(row)
            pb.update(1)
    return res


# ------------------------------ DIARIZATION ---------------------------------
def diarize_tx(audio, *, mode, cli=None, mdl=None, lang):
    patch_torchaudio()
    if not HUGGINGFACE_TOKEN:
        raise RuntimeError(
            "[FATAL] --diarize requested but HUGGINGFACE_TOKEN is not set.\n"
            "  1) Accept the model license: https://huggingface.co/pyannote/speaker_diarization-3.1\n"
            "  2) Create a token:          https://hf.co/settings/tokens\n"
            "  3) export HUGGINGFACE_TOKEN=hf_XXX"
        )
    from pyannote.audio import Pipeline  # import late

    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN
        )
    except Exception as e:
        raise RuntimeError(
            f"[FATAL] Could not load diarization pipeline:\n{e}"
        ) from None

    if pipe is None:
        raise RuntimeError("[FATAL] Diarization pipeline failed to load")

    # progress hook optional
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError:
        ProgressHook = None

    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio.export(wav, format="wav")
    device = prefer_device()
    pipe.to(device)
    log(f"[STEP] Diarization on {device}")

    try:
        if ProgressHook:
            with ProgressHook() as hook:
                diar = pipe(wav, hook=hook)
        else:
            with Spinner("[STEP] Diarization in progress"):
                diar = pipe(wav)
    finally:
        os.unlink(wav)

    limit = SECONDS_LIMIT * 1000
    full = audio
    jobs, spk_map, spk_idx, order = [], {}, 1, 0
    for turn, _, sp in diar.itertracks(yield_label=True):
        if turn.end - turn.start < 0.11:
            continue
        spk_map.setdefault(sp, f"Speaker {spk_idx}")
        spk_idx = len(spk_map) + 1
        segment = full[int(turn.start * 1000) : int(turn.end * 1000)]
        for off in range(0, len(segment), limit):
            part = segment[off : off + limit]
            st = turn.start + off / 1000
            ed = min(turn.end, st + len(part) / 1000)
            jobs.append((order, spk_map[sp], part, st, ed))
            order += 1

    res = [None] * len(jobs)

    def work(j):
        i, sp, pt, st, ed = j
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        pt.export(tmp, format="mp3", bitrate=f"{BITRATE}k")
        try:
            txt = (
                w_tx_api(cli, tmp, lang)
                if mode == "api"
                else w_tx_local(mdl, tmp, lang)
            )
        finally:
            os.unlink(tmp)
        return i, sp, txt, st, ed

    with tqdm(
        total=len(jobs), desc="[STEP] Transcribing segments", unit="seg", ncols=80
    ) as pb, ThreadPoolExecutor(MAX_CONC_REQUESTS if mode == "api" else 1) as pool:
        for fut in as_completed(pool.submit(work, j) for j in jobs):
            i, *row = fut.result()
            res[i] = tuple(row)
            pb.update(1)
    return res


# ------------------------------ POST-PROCESS --------------------------------
def merge(rows):
    if not rows:
        return rows
    m = [list(rows[0])]
    for sp, tx, st, ed in rows[1:]:
        sp0, tx0, _, ed0 = m[-1]
        if sp == sp0:
            m[-1][1] = f"{tx0} {tx}"
            m[-1][3] = ed
        else:
            m.append([sp, tx, st, ed])
    return [tuple(r) for r in m]


def to_lines(rows, show):
    return [
        f"{f'[{fmt_ts(st)}-{fmt_ts(ed)}] ' if show else ''}{f'[{sp}] ' if sp else ''}{tx}"
        for sp, tx, st, ed in rows
    ]


# ------------------------------ ORCHESTRATE ---------------------------------
def run(infile, outfile, *, mode, local_tag, diarize, lang, show_ts, agg):
    cli = mdl = None
    if mode == "api":
        if openai is None:
            raise RuntimeError("openai package missing.")
        if OPENAI_API_KEY == "your-api-key-here":
            raise RuntimeError("Set OPENAI_API_KEY")
        cli = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=httpx.Timeout(120, read=PER_REQ_TIMEOUT),
            max_retries=3,
        )
    else:
        mdl = load_local(local_tag, prefer_device())

    mp3 = convert(infile)
    cleaned = remove_silence(mp3)
    if mp3 != infile:
        os.unlink(mp3)
    rows = (
        diarize_tx(cleaned, mode=mode, cli=cli, mdl=mdl, lang=lang)
        if diarize
        else transcribe_chunks(chunk(cleaned), mode=mode, cli=cli, mdl=mdl, lang=lang)
    )
    if agg:
        rows = merge(rows)
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(to_lines(rows, show_ts)))
    log(f"[SUCCESS] Transcript → {os.path.abspath(outfile)}")


# ------------------------------ CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Transcribe audio via Whisper (API/local)."
    )
    ap.add_argument("input_file")
    ap.add_argument("output_file")
    ap.add_argument("--mode", choices=["api", "local"], default="api")
    ap.add_argument("--local-model", default="base")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--language", default=None, metavar="ISO")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    run(
        args.input_file,
        args.output_file,
        mode=args.mode,
        local_tag=args.local_model,
        diarize=args.diarize,
        lang=args.language,
        show_ts=args.timestamps,
        agg=args.aggregate,
    )


if __name__ == "__main__":
    log(f"[BOOT] {sys.argv}")
    main()
