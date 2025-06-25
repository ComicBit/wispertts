#!/usr/bin/env python3
"""
Transcribe an audio file with optional speaker diarization.

• FFmpeg progress bars (conversion + silence removal)
• Native pyannote ProgressHook for diarization progress
• Concurrency, retries, oversize-segment splitting
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
from typing import List

import httpx
import openai
import torch
from pydub import AudioSegment
from tqdm import tqdm

# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

MAX_CONC_REQUESTS      = 4             # < 5 000 Whisper-1 RPM
PER_REQUEST_TIMEOUT_S  = 90
MAX_WHISPER_MB         = 25
SAFETY_MARGIN          = 0.95
BITRATE_KBPS           = 128

TARGET_BYTES  = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE_KBPS / 8 * 1024))

# ---------------------------------------------------------------------------
def print_and_flush(msg: str):
    print(msg, flush=True)

# ---------------- Small UI helpers ------------------------------------------
class ProgressSpinner:
    """Fallback spinner for long steps when finer progress unavailable."""
    def __init__(self, desc="Working…"):
        self.desc, self._stop = desc, threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
    def __enter__(self):
        print_and_flush(self.desc)
        self._thread.start()
        return self
    def __exit__(self, *_):
        self._stop.set(); self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.desc) + 4) + "\r"); sys.stdout.flush()
    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r{self.desc} {ch}"); sys.stdout.flush(); time.sleep(0.1)

# ---------------- FFmpeg helpers --------------------------------------------
TIME_RE = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")

def _duration_sec(path: str) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            text=True).strip()
        return float(out)
    except Exception:
        return None

def _run_ffmpeg(cmd: List[str], total_s: float | None, desc: str):
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

# ---------------- Audio I/O --------------------------------------------------
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
    print_and_flush(f"[OK] Silence removed.")
    return audio

def seg_to_bytes(seg: AudioSegment) -> bytes:
    buf = BytesIO(); seg.export(buf, format="mp3", bitrate=f"{BITRATE_KBPS}k"); return buf.getvalue()

def chunk_audio(audio: AudioSegment) -> List[str]:
    total, limit_ms = len(audio), SECONDS_LIMIT * 1000
    paths, start = [], 0
    print_and_flush(f"[STEP] Chunking: {total/1000:.1f}s total, {SECONDS_LIMIT}s max/chunk")
    with tqdm(total=(total // limit_ms + 1), desc="[STEP] Chunking", unit="chunk", ncols=80) as bar:
        while start < total:
            end = min(start + limit_ms, total)
            if end < total and audio[end-2000:end].dBFS < -45: end -= 2000
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            audio[start:end].export(tmp, format="mp3", bitrate=f"{BITRATE_KBPS}k")
            paths.append(tmp); start = end; bar.update(1)
    return paths

# ---------------- OpenAI call wrapper ---------------------------------------
def _whisper_retry(
    client: openai.OpenAI, path: str, *,
    model="whisper-1", language=None,
    timeout_s=PER_REQUEST_TIMEOUT_S, tries=5, base_delay=2,
) -> str:
    for attempt in range(1, tries + 1):
        try:
            with open(path, "rb") as f:
                return client.audio.transcriptions.create(
                    model=model, file=f, response_format="text",
                    timeout=timeout_s, language=language).strip()
        except (openai.APITimeoutError, openai.RateLimitError) as e:
            delay = base_delay ** attempt
            print_and_flush(f"[WARN] {e.__class__.__name__} → retry {attempt}/{tries} in {delay}s"); time.sleep(delay)
    raise RuntimeError("Whisper failed after retries")

# ---------------- Parallel transcription ------------------------------------
def transcribe_paths(client, paths, *, model, language):
    texts = [None] * len(paths)
    def worker(i_p): i, p = i_p; return i, _whisper_retry(client, p, model=model, language=language)
    with tqdm(total=len(paths), desc="[STEP] Transcribing", unit="chunk", ncols=80) as bar, \
         ThreadPoolExecutor(MAX_CONC_REQUESTS) as pool:
        for fut in as_completed(pool.submit(worker, ip) for ip in enumerate(paths)):
            i, t = fut.result(); texts[i] = t; bar.update(1)
    return texts

# ---------------- Diarization pipeline --------------------------------------
def diarize_and_transcribe(audio: AudioSegment, out_txt: str, *, client, model, language):
    from pyannote.audio import Pipeline
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        Hook = ProgressHook
    except Exception:
        Hook = None  # older pyannote, fallback to spinner

    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio.export(wav, format="wav")

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=HUGGINGFACE_TOKEN)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pipe.to(device)

    print_and_flush("[STEP] Running speaker diarization")
    if Hook:
        with Hook() as hook:
            diar = pipe(wav, hook=hook)          # native tqdm bars
    else:
        with ProgressSpinner("[STEP] Diarization in progress"):
            diar = pipe(wav)

    # ----- split long turns --------------------------------------------------
    limit_ms = SECONDS_LIMIT * 1000
    audio_full = AudioSegment.from_wav(wav); os.unlink(wav)
    jobs, spk_map, spk_idx, order = [], {}, 1, 0
    for turn, _, spk in diar.itertracks(yield_label=True):
        if (turn.end - turn.start) < 0.11: continue
        spk_map.setdefault(spk, f"Speaker {spk_idx}"); spk_idx = len(spk_map) + 1
        seg = audio_full[int(turn.start*1000):int(turn.end*1000)]
        for part in [seg[i:i+limit_ms] for i in range(0, len(seg), limit_ms)]:
            jobs.append((order, spk_map[spk], part)); order += 1

    # ----- concurrent Whisper -----------------------------------------------
    results = [None] * len(jobs)
    def w(job):
        idx, spk, seg = job
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        seg.export(tmp, format="mp3", bitrate=f"{BITRATE_KBPS}k")
        try: txt = _whisper_retry(client, tmp, model=model, language=language)
        finally: os.unlink(tmp)
        return idx, spk, txt

    with tqdm(total=len(jobs), desc="[STEP] Transcribing segments", unit="segment", ncols=80) as bar, \
         ThreadPoolExecutor(MAX_CONC_REQUESTS) as pool:
        for fut in as_completed(pool.submit(w, j) for j in jobs):
            idx, spk, txt = fut.result(); results[idx] = (spk, txt); bar.update(1)

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"[{s}] {t}" for s, t in results if t))
    print_and_flush(f"[SUCCESS] Diarized transcript → {os.path.abspath(out_txt)}")
    return os.path.abspath(out_txt)

# ---------------- Orchestrator ----------------------------------------------
def transcribe_file(in_path, out_path="transcription.txt", *, diarize=False, language=None, model="whisper-1"):
    if OPENAI_API_KEY == "your-api-key-here": raise RuntimeError("Set OPENAI_API_KEY")
    client = openai.OpenAI(api_key=OPENAI_API_KEY,
                           timeout=httpx.Timeout(120, read=PER_REQUEST_TIMEOUT_S),
                           max_retries=3)
    mp3 = convert_to_mp3(in_path)
    cleaned = remove_silence(mp3)

    if diarize:
        if not HUGGINGFACE_TOKEN: raise RuntimeError("Set HUGGINGFACE_TOKEN for diarization")
        return diarize_and_transcribe(cleaned, out_path, client=client, model=model, language=language)

    chunks = chunk_audio(cleaned)
    texts = transcribe_paths(client, chunks, model=model, language=language)
    with open(out_path, "w", encoding="utf-8") as fh: fh.write("\n".join(texts))
    print_and_flush(f"[SUCCESS] Transcript → {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)

# ---------------- CLI --------------------------------------------------------
def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Transcribe audio with optional speaker diarization.")
    p.add_argument("input_file"); p.add_argument("output_file", nargs="?", default="transcription.txt")
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization (needs pyannote & HF token)")
    p.add_argument("--language", default=None, metavar="ISO_CODE",
                   help="Whisper language code, e.g. 'en', 'it'. Empty → auto-detect")
    args = p.parse_args(argv)
    try:
        transcribe_file(args.input_file, args.output_file,
                        diarize=args.diarize, language=args.language)
    except Exception as e:
        print_and_flush(f"[FATAL] {e}"); sys.exit(1)

if __name__ == "__main__":
    print_and_flush(f"[BOOT] {sys.argv}"); main()