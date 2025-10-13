#!/usr/bin/env python3
"""
Whisper transcription CLI

Features
--------
• MLX mode (default on Mac) – uses Lightning Whisper MLX for Apple Silicon optimization
• API mode             – uses OpenAI Whisper
• Local mode           – `--mode local --local-model medium` loads openai-whisper
• Automatic device     – MLX → CUDA → MPS → CPU, with Sparse-MPS → CPU fallback
• Optional diarization – `--diarize` (needs HuggingFace token & licence accept)
• --timestamps / --aggregate flags
• Robust FFmpeg progress, chunking, retries
"""

from __future__ import annotations
import argparse
import itertools
def ensure_mlx_download(tag: str, quant: Optional[str]):
    """Ensure the MLX Hugging Face repo for `tag` is downloaded into ./mlx_models.

    This tries to use `huggingface_hub.snapshot_download` when available. It's
    safe to call even if the MLX python packages are not installed; it simply
    attempts to populate the local `mlx_models/<model>` folder so MLX can use
    a local repo copy later.
    """
    base = MLX_MODEL_MAP.get(tag, tag)
    repo = globals().get("MLX_MODEL_REPOS", {}).get(base, f"mlx-community/whisper-{base}")
    local_target = os.path.join(os.path.dirname(__file__), "mlx_models", base)

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        log(f"[INFO] huggingface_hub not available; cannot pre-download MLX model '{base}'.")
        return

    # If a usable local model already exists, nothing to do.
    if os.path.isdir(local_target) and os.path.isfile(os.path.join(local_target, "config.json")):
        log(f"[INFO] MLX model '{base}' already present at {local_target}")
        return

    try:
        log(f"[INFO] Ensuring MLX model '{base}' is available: {repo} → {local_target}")
        snapshot_download(repo_id=repo, local_dir=local_target, local_dir_use_symlinks=False)
        if os.path.isfile(os.path.join(local_target, "config.json")):
            log(f"[INFO] MLX model '{base}' downloaded to {local_target}")
        else:
            log(f"[WARN] MLX model '{base}' download finished but config.json missing; MLX may still fetch from HF at runtime.")
    except Exception as e:
        log(f"[WARN] Could not pre-download MLX model '{repo}': {e}")


def install_lightning_if_supported() -> bool:
    """Try to install and import a Lightning MLX implementation if supported.

    Returns True if Lightning became available, False otherwise. This will
    attempt to pip-install `lightning-whisper-mlx` into the current Python
    interpreter (using `sys.executable`) only when running on Apple Silicon.
    """
    global LightningWhisperMLX, _MLX_BACKEND
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module("lightning_whisper_mlx")
        LightningWhisperMLX = getattr(mod, "LightningWhisperMLX", None) or mod.LightningWhisperMLX
        _MLX_BACKEND = "lightning"
        log("[INFO] lightning_whisper_mlx already importable")
        return True
    except Exception:
        # Not importable yet; consider installing.
        pass

    if not is_apple_silicon():
        log("[INFO] Automatic Lightning install skipped: not running on Apple Silicon")
        return False

    pkgs = ["lightning-whisper-mlx"]
    for pkg in pkgs:
        try:
            log(f"[INFO] Attempting to pip install {pkg} into {sys.executable}")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
            importlib.invalidate_caches()
            mod = importlib.import_module("lightning_whisper_mlx")
            LightningWhisperMLX = getattr(mod, "LightningWhisperMLX", None) or mod.LightningWhisperMLX
            _MLX_BACKEND = "lightning"
            log(f"[INFO] Successfully installed and imported {pkg}")
            return True
        except Exception as e:
            log(f"[WARN] Could not install {pkg}: {e}")

    return False

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

import httpx
import importlib
import torch
try:
    from pydub import AudioSegment
except ImportError as e:
    print(f"Warning: pydub import failed: {e}")
    print("Try installing: pip install pydub")
    AudioSegment = None
from tqdm import tqdm
import inspect
import soundfile as sf

# ------------------------------ ENV -----------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# ------------------------------ CONSTANTS -----------------------------------
MAX_CONC_REQUESTS, PER_REQ_TIMEOUT = 4, 90
MAX_MB, SAFETY, BITRATE = 25, 0.95, 128
TARGET_BYTES = int(MAX_MB * 1_048_576 * SAFETY)
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE / 8 * 1024))
WHISPER_CACHE = os.path.join(os.path.dirname(__file__), "models", "whisper")

# Regex used to parse ffmpeg/stderr progress lines like: time=00:01:23.45
T_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


# ------------------------------ HELPERS -------------------------------------
def log(msg: str):
    print(msg, flush=True)


def _emit_progress(stage: str, current: float, total: float | int | None, *, label: str | None = None,
                   detail: str | None = None, extra: dict | None = None):
    """Emit a structured progress event.

    Parameters
    ----------
    stage: str
        Canonical stage key (e.g. "convert", "transcribe").
    current: float
        Amount completed within the stage.
    total: float | int | None
        Total amount for the stage. If ``None`` or ``<= 0`` we treat it as ``1``.
    label: str, optional
        Human readable label for the current stage.
    detail: str, optional
        Secondary text that may be displayed in UI.
    extra: dict, optional
        Additional metadata (e.g. counts) to pass along.
    """

    if total is None or total <= 0:
        total = 1.0
    payload = {
        "stage": stage,
        "current": max(float(current), 0.0),
        "total": float(total),
    }
    if label:
        # Remove leading bracketed tokens like "[STEP]" or "[SUCCESS]" for UI
        try:
            lab = re.sub(r"^\[[^\]]+\]\s*", "", str(label)).strip()
        except Exception:
            lab = str(label)
        payload["label"] = lab
    if detail:
        payload["detail"] = detail
    if extra:
        payload["extra"] = extra
    try:
        log(f"[PROGRESS] {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}")
    except Exception:
        # Progress emission is best-effort; never break the pipeline.
        pass


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def prefer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_mode() -> str:
    """Get the default mode based on platform"""
    # Prefer MLX on Apple Silicon so we prepare the MLX backend first.
    # We do not attempt to import MLX here; the runtime will try to install
    # or fall back as needed after the user's choice is resolved.
    if is_apple_silicon():
        return "mlx"
    return "local"


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


def _run(cmd: list[str], total: Optional[float], desc: str, *, stage: str | None = None):
    if stage:
        _emit_progress(stage, 0, total or 1, label=desc)
    if not total:
        with tqdm(
            total=1, desc=desc, bar_format="{l_bar}{bar}|{elapsed}", ncols=80
        ) as pb:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
            pb.update(1)
        if stage:
            _emit_progress(stage, 1, 1, label=desc)
        return

    with tqdm(total=int(total), desc=desc, unit="s", ncols=80) as pb:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            if m := T_RE.search(line):
                h, mn, s = m.groups()
                pb.n = min(int(float(s) + int(mn) * 60 + int(h) * 3600), pb.total)
                pb.refresh()
                if stage:
                    _emit_progress(stage, pb.n, pb.total, label=desc)
        proc.wait()
        pb.n = pb.total
        pb.refresh()
        if stage:
            _emit_progress(stage, pb.total, pb.total, label=desc)
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
            data, sr = sf.read(path, always_2d=True, dtype="float32")
            return torch.from_numpy(data.T), sr
        return original(path, *args, **kwargs)

    torchaudio.load = load


def convert(src: str) -> str:
    """Return ``src`` if already MP3, otherwise convert and return new file."""

    if src.lower().endswith(".mp3"):
        return src

    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    _run(
        ["ffmpeg", "-y", "-i", src, dst],
        _dur(src),
        "Converting → MP3",
        stage="convert",
    )
    return dst


def remove_silence(p):
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    filt = (
        "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:"
        "stop_periods=-1:stop_duration=0.3:stop_threshold=-50dB"
    )
    _run(
        ["ffmpeg", "-y", "-i", p, "-af", filt, dst],
        _dur(p),
        "Removing silence",
        stage="silence",
    )
    audio = AudioSegment.from_file(dst)
    os.unlink(dst)
    return audio


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _estimate_chunks(audio) -> int:
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


def chunk(audio):
    lim = SECONDS_LIMIT * 1000
    total = len(audio)
    start = 0
    chunks = []
    with tqdm(
        total=_estimate_chunks(audio), desc="[STEP] Chunking", unit="chunk", ncols=80
    ) as pb:
        _emit_progress("chunk", 0, pb.total, label="[STEP] Chunking", extra={"chunks_total": pb.total})
        while start < total:
            end = min(start + lim, total)
            if end < total and audio[end - 2000 : end].dBFS < -45:
                end -= 2000
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            audio[start:end].export(tmp, format="mp3", bitrate=f"{BITRATE}k")
            chunks.append((tmp, start / 1000, end / 1000))
            start = end
            pb.update(1)
            _emit_progress(
                "chunk",
                pb.n,
                pb.total,
                label="[STEP] Chunking",
                extra={"chunks_total": pb.total, "chunks_done": pb.n},
            )
        if pb.total:
            _emit_progress(
                "chunk",
                pb.total,
                pb.total,
                label="[STEP] Chunking",
                extra={"chunks_total": pb.total, "chunks_done": pb.total},
            )
    return chunks


# ------------------------------ MLX WHISPER ---------------------------------
LightningWhisperMLX = None
_MLX_BACKEND = None

try:
    # Import Lightning Whisper MLX from the local folder if present.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lightning-whisper-mlx"))
    from lightning_whisper_mlx import LightningWhisperMLX as _LightningWhisperMLX

    LightningWhisperMLX = _LightningWhisperMLX
    _MLX_BACKEND = "lightning"
except ImportError:
    try:
        # Fallback to Apple's official mlx-whisper package when Lightning isn't vendored.
        from mlx_whisper import transcribe as _mlx_transcribe
        import mlx.core as mx

        MLX_MODEL_ROOT = os.path.join(os.path.dirname(__file__), "mlx_models")
        MLX_MODEL_REPOS = {
            "tiny": "mlx-community/whisper-tiny",
            "base": "mlx-community/whisper-base",
            "small": "mlx-community/whisper-small",
            "medium": "mlx-community/whisper-medium",
            "large": "mlx-community/whisper-large",
            "large-v2": "mlx-community/whisper-large-v2",
            # Note: MLX large-v3 repo uses the '-mlx' suffix on HF
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }

        def _resolve_mlx_repo(tag: str, quant: Optional[str]) -> str:
            base = MLX_MODEL_MAP.get(tag, tag)
            quant_suffix = (quant or "").lower().strip() or None
            search_paths = []
            if quant_suffix:
                search_paths.append(os.path.join(MLX_MODEL_ROOT, f"{base}-{quant_suffix}"))
            search_paths.append(os.path.join(MLX_MODEL_ROOT, base))
            for path in search_paths:
                if os.path.isdir(path):
                    # Ensure the MLX model directory contains a config.json at top-level.
                    cfg = os.path.join(path, "config.json")
                    if os.path.isfile(cfg):
                        return path
                    # If config is not present but there's a huggingface cache, the
                    # download may have started but not finished or files were
                    # left in a .cache subtree. Treat this as "not ready" and
                    # continue so we fall back to the HF repo string instead.
                    cache_dir = os.path.join(path, ".cache", "huggingface")
                    if os.path.isdir(cache_dir):
                        log(
                            f"[WARN] Found partial MLX model directory at {path} (contains .cache/huggingface) but missing top-level config.json.\n"
                            "  The model download may be incomplete. Remove the folder or finish the download to use a local MLX model, or omit the local MLX directory to fetch from the Hugging Face repo."
                        )
                    else:
                        log(f"[WARN] MLX model dir {path} is present but missing config.json")

            if quant_suffix:
                log(
                    f"[WARN] MLX quant='{quant}' weights not found locally – using full precision."
                )

            # No usable local model path found. Prefer to download a local copy
            # of the MLX HF repo into ./mlx_models/<base> so future runs don't
            # need to stream weights. If huggingface_hub is available, try to
            # snapshot_download the repo; otherwise return the HF repo string
            # so mlx_whisper can fetch it on-demand.
            repo = MLX_MODEL_REPOS.get(base, f"mlx-community/whisper-{base}")
            local_target = os.path.join(MLX_MODEL_ROOT, base)
            try:
                from huggingface_hub import snapshot_download

                log(f"[INFO] No local MLX model found for '{base}'. Attempting to download from {repo} → {local_target}")
                # snapshot_download will create the folder; set use symlinks off to ensure portability
                snapshot_download(repo_id=repo, local_dir=local_target, local_dir_use_symlinks=False)
                if os.path.isfile(os.path.join(local_target, "config.json")):
                    log(f"[INFO] MLX model '{base}' downloaded to {local_target}")
                    return local_target
                else:
                    log(f"[WARN] Downloaded {repo} but config.json still missing; will use remote repo string")
            except Exception as e:
                log(f"[WARN] Could not download MLX model '{repo}' locally: {e}. Will use remote HF repo instead.")

            return repo

        class LightningWhisperMLX:  # type: ignore[override]
            """Compatibility wrapper around mlx_whisper.transcribe."""

            def __init__(self, model: str = "base", batch_size: int = 12, quant: Optional[str] = None):
                self.model = model
                self.batch_size = batch_size
                self.quant = quant
                self.repo = _resolve_mlx_repo(model, quant)
                # Default to fp16 unless quantization requested (quant configs carry dtype).
                self.dtype = mx.float16 if quant is None else mx.float32

            def transcribe(self, audio_path: str, language: Optional[str] = None):
                decode_opts = {}
                if language:
                    decode_opts["language"] = language
                return _mlx_transcribe(
                    audio_path,
                    path_or_hf_repo=self.repo,
                    verbose=False,
                    fp16=self.dtype == mx.float16,
                    **decode_opts,
                )

        _MLX_BACKEND = "mlx_whisper"
    except ImportError:
        LightningWhisperMLX = None

MLX_BACKEND = _MLX_BACKEND

_MLX_MODEL_CACHE = {}

# Model mapping from standard names to MLX equivalents
MLX_MODEL_MAP = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
}

def load_mlx(tag: str, batch_size: int = 12, quant: Optional[str] = None):
    """Load Lightning Whisper MLX model with caching"""
    cache_key = f"{tag}_{batch_size}_{quant}"
    if cache_key in _MLX_MODEL_CACHE:
        return _MLX_MODEL_CACHE[cache_key]
        
    if LightningWhisperMLX is None:
        raise RuntimeError(
            "MLX mode requested but Lightning Whisper MLX is not available. "
            "Install MLX dependencies or use --mode local/api instead."
        )
    
    # Map standard model names to MLX names
    mlx_model = MLX_MODEL_MAP.get(tag, tag)
    
    try:
        label = f"Loading Lightning Whisper MLX '{mlx_model}'"
        _emit_progress("load_model", 0, 1, label=label, extra={"backend": "mlx", "batch_size": batch_size, "quant": quant})
        log(f"{label} (batch_size={batch_size}, quant={quant}) …")
        mdl = LightningWhisperMLX(model=mlx_model, batch_size=batch_size, quant=quant)
    except Exception as e:
        raise RuntimeError(f"Failed to load MLX model '{mlx_model}': {e}")
    
    _emit_progress("load_model", 1, 1, label=f"MLX model '{mlx_model}' ready", extra={"backend": "mlx"})
    _MLX_MODEL_CACHE[cache_key] = mdl
    return mdl


def w_tx_mlx(m, p, lang):
    """Transcribe using Lightning Whisper MLX"""
    try:
        result = m.transcribe(p, language=lang)
        return result.get('text', '').strip()
    except Exception as e:
        raise RuntimeError(f"MLX transcription failed: {e}")


# ------------------------------ LOCAL WHISPER -------------------------------
try:
    import whisper
except ImportError:
    whisper = None
_MODEL_CACHE = {}


class _SparseMPSFallback(Exception):
    """Signal that sparse MPS kernels are unavailable and MLX should be used."""

    def __init__(self, tag: str):
        self.tag = tag
        super().__init__(tag)


def load_local(tag: str, dev: torch.device):
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]
    if whisper is None:
        raise RuntimeError(
            "Local mode requested but `openai-whisper` is not installed."
        )
    try:
        label = f"Loading Whisper '{tag}' on {dev}"
        _emit_progress("load_model", 0, 1, label=label, extra={"backend": "local", "device": str(dev)})
        log(f"{label} …")
        mdl = whisper.load_model(tag, device=str(dev), download_root=WHISPER_CACHE)
    except (RuntimeError, NotImplementedError) as e:
        err = str(e)
        if "SparseMPS" in err or "_sparse_coo_tensor" in err:
            if LightningWhisperMLX is not None and is_apple_silicon():
                log("[WARN] Sparse op missing on MPS – switching to MLX backend")
                raise _SparseMPSFallback(tag) from e
            log("[WARN] Sparse op missing on MPS – falling back to CPU")
            mdl = whisper.load_model(tag, device="cpu", download_root=WHISPER_CACHE)
        else:
            raise
    _MODEL_CACHE[tag] = mdl
    _emit_progress("load_model", 1, 1, label=f"Local model '{tag}' ready", extra={"backend": "local", "device": str(dev)})
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
            if mode == "api":
                txt = w_tx_api(cli, p, lang)
            elif mode == "mlx":
                txt = w_tx_mlx(mdl, p, lang)
            else:  # local
                txt = w_tx_local(mdl, p, lang)
        finally:
            os.unlink(p)
        return i, None, txt, st, ed

    workers = MAX_CONC_REQUESTS if mode == "api" else 1
    workers = min(workers, len(chs)) or 1
    with tqdm(
        total=len(chs), desc="[STEP] Transcribing", unit="chunk", ncols=80
    ) as pb, ThreadPoolExecutor(workers) as pool:
        _emit_progress(
            "transcribe",
            0,
            pb.total or 1,
            label="Transcribing audio",
            extra={"units_total": pb.total, "units_done": 0},
        )
        for fut in as_completed(pool.submit(work, x) for x in enumerate(chs)):
            i, *row = fut.result()
            res[i] = tuple(row)
            pb.update(1)
            _emit_progress(
                "transcribe",
                pb.n,
                pb.total or 1,
                label="Transcribing audio",
                extra={"units_total": pb.total, "units_done": pb.n},
            )
        if pb.total:
            _emit_progress(
                "transcribe",
                pb.total,
                pb.total,
                label="Transcribing audio",
                extra={"units_total": pb.total, "units_done": pb.total},
            )
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

    buf = BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    data, sr = sf.read(buf, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T)

    device = prefer_device()
    pipe.to(device)
    log(f"Diarization on {device}")
    _emit_progress("diarize", 0, 1, label="Running diarization")

    item = {"waveform": waveform, "sample_rate": sr}
    try:
        if ProgressHook:
            with ProgressHook() as hook:
                diar = pipe(item, hook=hook)
        else:
            with Spinner("Diarization in progress"):
                diar = pipe(item)
    finally:
        buf.close()

    limit_ms = int(SECONDS_LIMIT * 1000)
    full = audio
    if not isinstance(full.sample_width, int):
        full = full.set_sample_width(int(round(full.sample_width)))
    if not isinstance(full.frame_rate, int):
        full = full.set_frame_rate(int(round(full.frame_rate)))
    frame_rate = max(int(round(full.frame_rate)), 1)
    samples_per_ms = frame_rate / 1000.0
    limit_frames = max(int(round(limit_ms * samples_per_ms)), frame_rate)
    jobs, spk_map, spk_idx, order = [], {}, 1, 0
    for turn, _, sp in diar.itertracks(yield_label=True):
        if turn.end - turn.start < 0.11:
            continue
        spk_map.setdefault(sp, f"Speaker {spk_idx}")
        spk_idx = len(spk_map) + 1
        start_frame = max(0, int(round(turn.start * frame_rate)))
        end_frame = max(start_frame + 1, int(round(turn.end * frame_rate)))
        for frame_off in range(start_frame, end_frame, limit_frames):
            frame_end = min(end_frame, frame_off + limit_frames)
            if frame_end <= frame_off:
                continue
            part = full.get_sample_slice(frame_off, frame_end)
            st = frame_off / frame_rate
            ed = frame_end / frame_rate
            jobs.append((order, spk_map[sp], part, st, ed))
            order += 1

    res = [None] * len(jobs)

    _emit_progress(
        "diarize",
        1,
        1,
        label="Diarization complete",
        extra={"jobs": len(jobs)},
    )

    def work(j):
        i, sp, pt, st, ed = j
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        pt.export(tmp, format="mp3", bitrate=f"{BITRATE}k")
        try:
            if mode == "api":
                txt = w_tx_api(cli, tmp, lang)
            elif mode == "mlx":
                txt = w_tx_mlx(mdl, tmp, lang)
            else:  # local
                txt = w_tx_local(mdl, tmp, lang)
        finally:
            os.unlink(tmp)
        return i, sp, txt, st, ed

    with tqdm(
        total=len(jobs), desc="[STEP] Transcribing segments", unit="seg", ncols=80
    ) as pb, ThreadPoolExecutor(MAX_CONC_REQUESTS if mode == "api" else 1) as pool:
        _emit_progress(
            "transcribe_segments",
            0,
            pb.total or 1,
            label="Transcribing diarized segments",
            extra={"units_total": pb.total, "units_done": 0},
        )
        for fut in as_completed(pool.submit(work, j) for j in jobs):
            i, *row = fut.result()
            res[i] = tuple(row)
            pb.update(1)
            _emit_progress(
                "transcribe_segments",
                pb.n,
                pb.total or 1,
                label="Transcribing diarized segments",
                extra={"units_total": pb.total, "units_done": pb.n},
            )
        if pb.total:
            _emit_progress(
                "transcribe_segments",
                pb.total,
                pb.total,
                label="Transcribing diarized segments",
                extra={"units_total": pb.total, "units_done": pb.total},
            )
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
def run(
    infile,
    outfile,
    *,
    mode,
    local_tag,
    mlx_batch_size,
    mlx_quant,
    diarize,
    lang,
    show_ts,
    agg,
    trim_silence=True,
    explicit_mode: bool = False,
):
    cli = mdl = None
    _emit_progress("prepare", 0, 1, label="Preparing transcription job")
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
        _emit_progress("load_model", 1, 1, label="Whisper API ready", extra={"backend": "api"})
    elif mode == "mlx":
        mdl = load_mlx(local_tag, batch_size=mlx_batch_size, quant=mlx_quant)
    else:  # local
        # If MLX is available on the system and the user did not explicitly
        # request local mode, prefer MLX and avoid calling `whisper.load_model`
        # which would trigger a standard .pt download.
        if LightningWhisperMLX is not None and not explicit_mode:
            log("[INFO] MLX backend present; using MLX instead of downloading standard Whisper .pt")
            mode = "mlx"
            mdl = load_mlx(local_tag, batch_size=mlx_batch_size, quant=mlx_quant)
        else:
            try:
                mdl = load_local(local_tag, prefer_device())
            except _SparseMPSFallback as err:
                # Sparse MPS kernels are missing. Prefer MLX on Apple Silicon when
                # available, but if the MLX model repository isn't present / is
                # incomplete (for example missing `config.json`), fall back to a
                # local CPU Whisper model so the user's downloaded `.pt` file can
                # still be used.
                if LightningWhisperMLX is None:
                    raise RuntimeError(
                        "Sparse MPS kernels missing and no MLX backend available; install 'mlx'"
                    ) from err

                # Try loading MLX first; if it raises (e.g., missing config.json
                # in `mlx_models/<model>`), fall back to CPU local Whisper.
                try:
                    mode = "mlx"
                    log("[INFO] Using MLX backend for local transcription")
                    mdl = load_mlx(local_tag, batch_size=mlx_batch_size, quant=mlx_quant)
                except Exception as e:
                    log(f"[WARN] MLX model load failed: {e} — falling back to CPU local Whisper")
                    mode = "local"
                    # Load local model on CPU so the existing models/whisper/<tag>.pt
                    # file (if present) is used instead of requiring an MLX repo.
                    mdl = load_local(local_tag, torch.device("cpu"))

    _emit_progress("prepare", 1, 1, label="Transcription pipeline ready")
    mp3 = convert(infile)
    if trim_silence:
        cleaned = remove_silence(mp3)
    else:
        cleaned = AudioSegment.from_file(mp3)
    if mp3 != infile:
        os.unlink(mp3)
    rows = (
        diarize_tx(cleaned, mode=mode, cli=cli, mdl=mdl, lang=lang)
        if diarize
        else transcribe_chunks(chunk(cleaned), mode=mode, cli=cli, mdl=mdl, lang=lang)
    )
    if agg:
        rows = merge(rows)
    _emit_progress("finalize", 0, 1, label="Finalizing transcript")
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(to_lines(rows, show_ts)))
    log(f"[SUCCESS] Transcript → {os.path.abspath(outfile)}")
    _emit_progress("finalize", 1, 1, label="Transcript ready")


# ------------------------------ CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Transcribe audio via Whisper (MLX/API/local)."
    )
    ap.add_argument("input_file")
    ap.add_argument("output_file")
    ap.add_argument("--mode", choices=["mlx", "api", "local"], default=None,
                    help="Transcription mode. Defaults to 'mlx' on Apple Silicon, 'api' otherwise.")
    ap.add_argument("--local-model", default="base", 
                    help="Model size for local/mlx modes")
    ap.add_argument("--mlx-batch-size", type=int, default=12,
                    help="Batch size for MLX mode (higher = faster but more memory)")
    ap.add_argument("--mlx-quant", choices=[None, "4bit", "8bit"], default=None,
                    help="Quantization for MLX mode (reduces memory usage)")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--language", default=None, metavar="ISO")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument(
        "--skip-silence",
        action="store_true",
        help="Disable silence removal pre-processing",
    )
    args = ap.parse_args()
    
    # Set default mode based on platform if not specified
    mode = args.mode or get_default_mode()
    
    # Validate MLX availability if requested
    if mode == "mlx" and not is_apple_silicon():
        log("[WARN] MLX mode requested but not running on Apple Silicon. Falling back to API mode.")
        mode = "api"
    
    # If the user explicitly asked for MLX, try to pre-download the HF repo
    # into ./mlx_models so that the runtime has the expected MLX repo layout
    # (config.json, weights.npz, etc.) available before any backend fallbacks.
    if mode == "mlx":
        try:
            ensure_mlx_download(args.local_model, args.mlx_quant)
        except Exception as e:
            log(f"[WARN] Pre-download attempt for MLX model failed: {e}")
    
    # If MLX is requested but we don't have Lightning available yet, try to
    # install it automatically (only on Apple Silicon) before falling back.
    if mode == "mlx" and LightningWhisperMLX is None:
        try:
            if install_lightning_if_supported():
                log("[INFO] Lightning backend installed and available; continuing with MLX mode")
        except Exception as e:
            log(f"[WARN] Automatic Lightning install attempt failed: {e}")

    if mode == "mlx" and LightningWhisperMLX is None:
        log("[WARN] MLX mode requested but Lightning Whisper MLX not available. Falling back to local mode.")
        mode = "local"
    
    run(
        args.input_file,
        args.output_file,
        mode=mode,
        local_tag=args.local_model,
        mlx_batch_size=args.mlx_batch_size,
        mlx_quant=args.mlx_quant,
        diarize=args.diarize,
        lang=args.language,
        show_ts=args.timestamps,
        agg=args.aggregate,
        trim_silence=not getattr(args, "skip_silence", False),
        explicit_mode=(args.mode is not None),
    )


if __name__ == "__main__":
    log(f"[BOOT] {sys.argv}")
    main()
