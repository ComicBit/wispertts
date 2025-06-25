#!/usr/bin/env python3
"""
Transcribe an audio file with optional speaker diarization.

New in this version
-------------------
• Automatic PyTorch device pick-up (CUDA ➜ MPS ➜ CPU)

Existing features
-----------------
• --timestamps and --aggregate flags
• FFmpeg progress bars, pyannote.ProgressHook for diarization
• Robust concurrency, retries, oversize-segment splitting
• --language ISO-code forwarded to Whisper
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Optional, Tuple

import httpx
import openai
import torch
from pydub import AudioSegment
from tqdm import tqdm

# ---------------------------------------------------------------------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

MAX_CONC_REQUESTS     = 4
PER_REQUEST_TIMEOUT_S = 90
MAX_WHISPER_MB        = 25
SAFETY_MARGIN         = 0.95
BITRATE_KBPS          = 128

TARGET_BYTES  = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE_KBPS / 8 * 1024))

# ---------------------------------------------------------------------------
def print_and_flush(msg: str):
    print(msg, flush=True)

# ---------------- Device helper ---------------------------------------------
def select_device() -> torch.device:
    """CUDA ➜ MPS ➜ CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------- Small UI helpers ------------------------------------------
class ProgressSpinner:
    def __init__(self, desc="Working…"):
        self.desc, self._stop = desc, threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
    def __enter__(self):
        print_and_flush(self.desc)
        self._thread.start(); return self
    def __exit__(self, *_):
        self._stop.set(); self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.desc) + 4) + "\r"); sys.stdout.flush()
    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r{self.desc} {ch}"); sys.stdout.flush(); time.sleep(0.1)

# ---------------- FFmpeg helpers --------------------------------------------
TIME_RE = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")

def _duration_sec(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            text=True
        ).strip()
        return float(out)
    except Exception:
        return None

def _run_ffmpeg(cmd: List[str], total_s: Optional[float], desc: str):
    if not total_s:
        with tqdm(total=1, desc=desc, bar_format='{l_bar}{bar}|{elapsed}', ncols=80) as pbar:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
            pbar.update(1)
        return
    with tqdm(total=int(total_s), desc=desc, unit="s", ncols=80) as pbar:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            m = TIME_RE.search(line)
            if m:
                h, m_, s = m.groups()
                cur = int(float(s) + int(m_) * 60 + int(h) * 3600)
                pbar.n = min(cur, pbar.total); pbar.refresh()
        proc.wait(); pbar.n = pbar.total; pbar.refresh()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

# ---------------- Audio helpers ---------------------------------------------
def convert_to_mp3(src: str) -> str:
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    _run_ffmpeg(["ffmpeg", "-y", "-i", src, dst], _duration_sec(src), "[STEP] Converting → MP3")
    print_and_flush(f"[OK] MP3 created: {dst}")
    return dst

def remove_silence(path: str) -> AudioSegment:
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    filt = ("silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:"
            "stop_periods=-1:stop_duration=0.3:stop_threshold=-50dB")
    _run_ffmpeg(["ffmpeg", "-y", "-i", path, "-af", filt, dst],
                _duration_sec(path), "[STEP] Removing silence")
    audio = AudioSegment.from_file(dst); os.unlink(dst)
    print_and_flush("[OK] Silence removed.")
    return audio

def format_ts(seconds: float) -> str:
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def chunk_audio(audio: AudioSegment) -> List[Tuple[str, float, float]]:
    total, limit_ms = len(audio), SECONDS_LIMIT * 1000
    chunks, start_ms = [], 0
    print_and_flush(f"[STEP] Chunking: {total/1000:.1f}s total, {SECONDS_LIMIT}s max/chunk")
    with tqdm(total=(total // limit_ms + 1), desc="[STEP] Chunking", unit="chunk", ncols=80) as bar:
        while start_ms < total:
            end_ms = min(start_ms + limit_ms, total)
            if end_ms < total and audio[end_ms-2000:end_ms].dBFS < -45:
                end_ms -= 2000
            seg = audio[start_ms:end_ms]
            path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            seg.export(path, format="mp3", bitrate=f"{BITRATE_KBPS}k")
            chunks.append((path, start_ms / 1000, end_ms / 1000))
            start_ms = end_ms; bar.update(1)
    return chunks

# ---------------- OpenAI call with retry ------------------------------------
def _whisper_retry(client: openai.OpenAI, path: str, *,
                   model="whisper-1", language=None, timeout_s=PER_REQUEST_TIMEOUT_S,
                   tries=5, base_delay=2) -> str:
    for attempt in range(1, tries + 1):
        try:
            with open(path, "rb") as f:
                return client.audio.transcriptions.create(
                    model=model, file=f, response_format="text",
                    timeout=timeout_s, language=language).strip()
        except (openai.APITimeoutError, openai.RateLimitError) as e:
            delay = base_delay ** attempt
            print_and_flush(f"[WARN] {e.__class__.__name__} → retry {attempt}/{tries} in {delay}s")
            time.sleep(delay)
    raise RuntimeError("Whisper failed after retries")

# ---------------- Transcription helpers -------------------------------------
def transcribe_chunks(client, chunks, *, model, language):
    results = [None] * len(chunks)
    def worker(i_c):
        i,(p,st,et) = i_c; txt = _whisper_retry(client,p,model=model,language=language)
        print_and_flush(f"[OK] Chunk {i+1}/{len(chunks)} transcribed"); return i,None,txt,st,et
    with tqdm(total=len(chunks), desc="[STEP] Transcribing", unit="chunk", ncols=80) as bar, \
         ThreadPoolExecutor(MAX_CONC_REQUESTS) as pool:
        for fut in as_completed(pool.submit(worker, ic) for ic in enumerate(chunks)):
            i,*row = fut.result(); results[i] = tuple(row); bar.update(1)
    return results  # [(None, text, start, end)]

# ---------------- Diarization + transcription -------------------------------
def diarize_and_transcribe(cleaned_audio, *, client, model, language):
    from pyannote.audio import Pipeline
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        Hook = ProgressHook
    except Exception:
        Hook = None

    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cleaned_audio.export(wav, format="wav")

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=HUGGINGFACE_TOKEN)
    device = select_device()
    print_and_flush(f"[STEP] Running speaker diarization on device: {device}")
    pipe.to(device)

    if Hook:
        with Hook() as hook: diar = pipe(wav, hook=hook)
    else:
        with ProgressSpinner("[STEP] Diarization in progress"): diar = pipe(wav)

    audio_full = AudioSegment.from_wav(wav); os.unlink(wav)
    limit_ms = SECONDS_LIMIT * 1000
    jobs, spk_map, spk_idx, order = [], {}, 1, 0
    for turn, _, spk in diar.itertracks(yield_label=True):
        if (turn.end - turn.start) < 0.11: continue
        spk_map.setdefault(spk, f"Speaker {spk_idx}"); spk_idx = len(spk_map) + 1
        seg = audio_full[int(turn.start*1000):int(turn.end*1000)]
        for off in range(0, len(seg), limit_ms):
            part = seg[off:off+limit_ms]
            st = turn.start + off/1000; et = min(turn.end, st+len(part)/1000)
            jobs.append((order, spk_map[spk], part, st, et)); order += 1

    results = [None]*len(jobs)
    def worker(j):
        idx,speaker,part,st,et = j
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        part.export(tmp, format="mp3", bitrate=f"{BITRATE_KBPS}k")
        try: txt = _whisper_retry(client,tmp,model=model,language=language)
        finally: os.unlink(tmp)
        return idx,speaker,txt,st,et

    with tqdm(total=len(jobs), desc="[STEP] Transcribing segments",
              unit="segment", ncols=80) as bar, \
         ThreadPoolExecutor(MAX_CONC_REQUESTS) as pool:
        for fut in as_completed(pool.submit(worker,j) for j in jobs):
            idx,*row = fut.result(); results[idx] = tuple(row); bar.update(1)
    return results  # [(spk, txt, start, end)]

# ---------------- Aggregation & formatting ----------------------------------
def merge_consecutive(rows):
    if not rows: return rows
    merged = [list(rows[0])]
    for spk,txt,st,et in rows[1:]:
        p_spk,p_txt,p_st,p_et = merged[-1]
        if spk == p_spk:
            merged[-1][1] = f"{p_txt} {txt}"
            merged[-1][3] = et
        else: merged.append([spk,txt,st,et])
    return [tuple(r) for r in merged]

def build_lines(rows, *, show_ts):
    out=[]
    for spk,txt,st,et in rows:
        ts = f"[{format_ts(st)}-{format_ts(et)}] " if show_ts else ""
        out.append(f"{ts}[{spk}] {txt}" if spk else f"{ts}{txt}")
    return out

# ---------------- Orchestrator ----------------------------------------------
def transcribe_file(input_path, output_txt="transcription.txt", *,
                    diarize=False, language=None, model="whisper-1",
                    show_timestamps=False, aggregate_lines=False):
    if OPENAI_API_KEY == "your-api-key-here": raise RuntimeError("Set OPENAI_API_KEY")

    client = openai.OpenAI(api_key=OPENAI_API_KEY,
                           timeout=httpx.Timeout(120, read=PER_REQUEST_TIMEOUT_S),
                           max_retries=3)

    mp3     = convert_to_mp3(input_path)
    cleaned = remove_silence(mp3)

    if diarize:
        rows = diarize_and_transcribe(cleaned_audio=cleaned,
                                      client=client, model=model, language=language)
    else:
        chunks = chunk_audio(cleaned)
        rows   = transcribe_chunks(client, chunks, model=model, language=language)

    if aggregate_lines: rows = merge_consecutive(rows)
    lines = build_lines(rows, show_ts=show_timestamps)

    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print_and_flush(f"[SUCCESS] Transcript → {os.path.abspath(output_txt)}")
    return os.path.abspath(output_txt)

# ---------------- CLI --------------------------------------------------------
def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Transcribe audio with optional speaker diarization.")
    p.add_argument("input_file"); p.add_argument("output_file", nargs="?", default="transcription.txt")
    p.add_argument("--diarize",    action="store_true", help="Enable speaker diarization (needs pyannote & HF token)")
    p.add_argument("--language",   default=None, metavar="ISO",
                   help="Whisper language code (e.g. 'en', 'it'). Empty → auto-detect")
    p.add_argument("--timestamps", action="store_true", help="Include [HH:MM:SS-HH:MM:SS] per line")
    p.add_argument("--aggregate",  action="store_true", help="Merge consecutive lines from same speaker")
    args = p.parse_args(argv)

    try:
        transcribe_file(args.input_file, args.output_file,
                        diarize=args.diarize, language=args.language,
                        show_timestamps=args.timestamps, aggregate_lines=args.aggregate)
    except Exception as e:
        print_and_flush(f"[FATAL] {e}"); sys.exit(1)

if __name__ == "__main__":
    print_and_flush(f"[BOOT] {sys.argv}"); main()