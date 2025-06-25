#!/usr/bin/env python3
"""
Transcribe an audio file with optional speaker diarization.

Key features
------------
* Converts any input format to MP3 → removes silence → splits into Whisper-size chunks
* Optional diarization with `pyannote/speaker-diarization-3.1`
* Concurrency with a single OpenAI client (thread-safe) and configurable worker pool
* Robust timeouts + retries for flaky networks / transient rate limits
* `--language` argument passes ISO-639-1/2 code to Whisper (leave unset for auto-detect)

Example
-------
python script.py meeting.wav transcript.txt --diarize --language it
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List

import httpx
import openai
import torch
from pydub import AudioSegment
from tqdm import tqdm

# ---------------- Configuration ---------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

MAX_CONCURRENT_REQUESTS = 4           # thread-pool size (safe: << 5 000 RPM limit)
PER_REQUEST_TIMEOUT_S   = 90          # Whisper call hard ceiling
MAX_WHISPER_MB          = 25          # API limit
SAFETY_MARGIN           = 0.95
BITRATE_KBPS            = 128

TARGET_BYTES   = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
SECONDS_LIMIT  = int(TARGET_BYTES / (BITRATE_KBPS / 8 * 1024))

# ---------------- Helpers ----------------------------------------------------
def print_and_flush(msg: str):
    print(msg, flush=True)


class Spinner:
    """Tiny tqdm spinner for quick, blocking steps."""
    def __init__(self, desc="Working…"):
        self.desc, self.pbar = desc, None
    def __enter__(self):
        self.pbar = tqdm(total=1, desc=self.desc, bar_format='{l_bar}{bar} {desc}', leave=False)
        return self
    def __exit__(self, *_):
        self.pbar.update(1)
        self.pbar.close()


def seg_to_bytes(seg: AudioSegment, fmt="mp3") -> bytes:
    buf = BytesIO()
    seg.export(buf, format=fmt, bitrate=f"{BITRATE_KBPS}k")
    return buf.getvalue()


# ---------------- Audio conversion / pre-processing --------------------------
def convert_to_mp3(input_path: str) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with Spinner("[STEP] Converting to MP3…"):
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", input_path, out], check=True)
    print_and_flush(f"[OK] ffmpeg MP3 created: {out}")
    return out


def remove_silence(path: str) -> AudioSegment:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    filt = ("silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:"
            "stop_periods=-1:stop_duration=0.3:stop_threshold=-50dB")
    with Spinner("[STEP] Removing silence…"):
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-af", filt, out], check=True)
    cleaned = AudioSegment.from_file(out)
    print_and_flush(f"[OK] Silence-removed MP3 ready: {out}")
    os.unlink(out)
    return cleaned


def chunk_audio(audio: AudioSegment) -> List[str]:
    total_ms, limit_ms = len(audio), SECONDS_LIMIT * 1000
    paths, start, idx = [], 0, 1
    print_and_flush(f"[STEP] Chunking audio: {total_ms/1000:.1f}s total, {SECONDS_LIMIT}s per chunk max")
    with tqdm(total=max(1, total_ms // limit_ms + 1), desc="[STEP] Chunking", unit="chunk") as pbar:
        while start < total_ms:
            end = min(start + limit_ms, total_ms)
            if end < total_ms:                           # back-track ~2 s to quiet spot
                snippet = audio[end-2000:end]
                if snippet.dBFS < -45:
                    end -= 2000
            seg = audio[start:end]
            data = seg_to_bytes(seg)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            with open(tmp, "wb") as f:
                f.write(data)
            print_and_flush(f"[INFO]   Chunk {idx}: {len(seg)/1000:.1f}s, {len(data)/1_048_576:.2f} MB")
            paths.append(tmp)
            start, idx = end, idx + 1
            pbar.update(1)
    print_and_flush(f"[OK] Chunking complete. {len(paths)} chunk(s) ready.")
    return paths


# ---------------- OpenAI call wrapper ---------------------------------------
def _whisper_with_retry(
    client: openai.OpenAI,
    file_path: str,
    *,
    model: str = "whisper-1",
    language: str | None = None,
    timeout_s: int = PER_REQUEST_TIMEOUT_S,
    max_attempts: int = 5,
    base_delay: int = 2,
) -> str:
    """Call Whisper with timeout + exponential back-off. Returns plain text."""
    for attempt in range(1, max_attempts + 1):
        try:
            with open(file_path, "rb") as f:
                return client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="text",
                    timeout=timeout_s,
                    language=language,
                ).strip()
        except (openai.APITimeoutError, openai.RateLimitError) as e:
            delay = base_delay ** attempt
            print_and_flush(f"[WARN] {e.__class__.__name__}: retry {attempt}/{max_attempts} in {delay}s")
            time.sleep(delay)
        except Exception:
            raise
    raise RuntimeError(f"Whisper failed after {max_attempts} attempts")


# ---------------- Parallel transcription (chunks) ---------------------------
def transcribe_paths(
    client: openai.OpenAI,
    paths: List[str],
    *,
    model: str,
    language: str | None,
) -> List[str]:
    texts = [None] * len(paths)

    def worker(i_p):
        i, path = i_p
        txt = _whisper_with_retry(client, path, model=model, language=language)
        print_and_flush(f"[OK] Transcribed chunk {i + 1}/{len(paths)}")
        return i, txt

    with tqdm(total=len(paths), desc="[STEP] Transcribing", unit="chunk") as pbar, \
         ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as exe:
        for fut in as_completed([exe.submit(worker, (idx, p)) for idx, p in enumerate(paths)]):
            i, t = fut.result()
            texts[i] = t
            pbar.update(1)

    print_and_flush("[OK] All chunks transcribed.")
    return texts


# ---------------- Diarization workflow --------------------------------------
def diarize_and_transcribe(
    cleaned_audio: AudioSegment,
    output_txt: str,
    *,
    client: openai.OpenAI,
    model: str,
    language: str | None,
) -> str:
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError("pyannote.audio is required for diarization. Install with `pip install pyannote.audio`")

    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cleaned_audio.export(wav_path, format="wav")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print_and_flush(f"[STEP] Running speaker diarization on device: {device}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    ).to(device)

    diarization = pipeline(wav_path)
    audio = AudioSegment.from_wav(wav_path)

    limit_ms = SECONDS_LIMIT * 1000
    jobs, speaker_map, speaker_counter = [], {}, 1
    order_counter = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg_len = turn.end - turn.start
        if seg_len < 0.11:
            continue
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"; speaker_counter += 1
        spk_name = speaker_map[speaker]
        seg = audio[int(turn.start * 1000):int(turn.end * 1000)]
        for part in [seg[i:i + limit_ms] for i in range(0, len(seg), limit_ms)]:
            jobs.append((order_counter, spk_name, part))
            order_counter += 1

    results = [None] * len(jobs)

    def worker(job):
        idx, spk, seg = job
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        seg.export(tmp, format="mp3")
        try:
            text = _whisper_with_retry(client, tmp, model=model, language=language)
        finally:
            os.unlink(tmp)
        return idx, spk, text

    with tqdm(total=len(jobs), desc="[STEP] Transcribing diarized segments", unit="segment") as pbar, \
         ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as exe:
        for fut in as_completed([exe.submit(worker, j) for j in jobs]):
            idx, spk, txt = fut.result()
            results[idx] = (spk, txt)
            pbar.update(1)

    lines = [f"[{spk}] {txt}" for spk, txt in results if txt]
    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print_and_flush(f"[SUCCESS] Written diarized transcription to: {os.path.abspath(output_txt)}")
    os.unlink(wav_path)
    return os.path.abspath(output_txt)


# ---------------- Top-level orchestration -----------------------------------
def transcribe_file(
    input_path: str,
    output_txt: str = "transcription.txt",
    *,
    diarize: bool = False,
    language: str | None = None,
    model: str = "whisper-1",
) -> str:
    print_and_flush(f"[ALIVE] Script launched. Transcribing: {input_path}")

    if OPENAI_API_KEY == "your-api-key-here":
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")

    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=httpx.Timeout(120.0, read=PER_REQUEST_TIMEOUT_S),
        max_retries=3,
    )
    print_and_flush("[OK] OpenAI client initialized.")

    input_path = convert_to_mp3(input_path)
    cleaned = remove_silence(input_path)

    if diarize:
        if not HUGGINGFACE_TOKEN:
            raise RuntimeError("Set HUGGINGFACE_TOKEN to accept model licenses.")
        return diarize_and_transcribe(
            cleaned, output_txt, client=client, model=model, language=language
        )

    paths = chunk_audio(cleaned)
    texts = transcribe_paths(client, paths, model=model, language=language)
    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(texts))
    print_and_flush(f"[SUCCESS] Finished! Transcription written to: {os.path.abspath(output_txt)}")
    return os.path.abspath(output_txt)


# ---------------- CLI entry-point -------------------------------------------
def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker diarization.")
    parser.add_argument("input_file", help="Input audio file (wav/mp3/etc.)")
    parser.add_argument("output_file", nargs="?", default="transcription.txt", help="Output text file")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker identification/diarization (requires pyannote.audio & HF token)")
    parser.add_argument("--language", metavar="ISO_CODE", default=None,
                        help="Language code for Whisper (e.g. 'en', 'it'). If omitted, auto-detect.")
    args = parser.parse_args(argv)

    try:
        transcribe_file(
            args.input_file,
            args.output_file,
            diarize=args.diarize,
            language=args.language,
        )
    except Exception as e:
        print_and_flush(f"[FATAL ERROR] Script exited with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print_and_flush(f"[BOOT] Script running as __main__ with args: {sys.argv}")
    main()