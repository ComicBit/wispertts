#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
from io import BytesIO
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import torch
from pydub import AudioSegment
from tqdm import tqdm
import httpx, time

# -------- Configuration --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Whisper API file-size limit
MAX_WHISPER_MB = 25
SAFETY_MARGIN = 0.95
TARGET_BYTES = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
BITRATE_KBPS = 128
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE_KBPS / 8 * 1024))
MAX_CONCURRENT_REQUESTS = 4
PER_REQUEST_TIMEOUT_S   = 90


def print_and_flush(msg):
    print(msg, flush=True)


# Spinner for quick steps
class Spinner:
    def __init__(self, desc="Working…"):
        self.desc = desc
        self.pbar = None
    def __enter__(self):
        self.pbar = tqdm(total=1, desc=self.desc, bar_format='{l_bar}{bar} {desc}', leave=False)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.update(1)
        self.pbar.close()


# -------- Convert Audio --------
def convert_to_mp3(input_path: str) -> str:
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with Spinner(desc="[STEP] Converting to MP3…"):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
            output_path
        ], check=True)
    print_and_flush(f"[OK] ffmpeg MP3 created: {output_path}")
    return output_path


# -------- Helpers --------
def seg_to_bytes(seg: AudioSegment, fmt="mp3") -> bytes:
    buf = BytesIO()
    seg.export(buf, format=fmt, bitrate=f"{BITRATE_KBPS}k")
    return buf.getvalue()

def _whisper_with_retry(
    client: openai.OpenAI,
    file_path: str,
    model: str = "whisper-1",
    timeout_s: int = PER_REQUEST_TIMEOUT_S,
    max_attempts: int = 5,
    base_delay: int = 2,
) -> str:
    """
    Call Whisper with a per-request timeout and exponential back-off.
    Returns plain text (can be empty). Raises on final failure.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            with open(file_path, "rb") as f:
                return client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="text",
                    timeout=timeout_s,        # overrides client default
                ).strip()
        except (openai.APITimeoutError, openai.RateLimitError) as e:
            delay = base_delay ** attempt
            print_and_flush(f"[WARN] {e.__class__.__name__}: retry {attempt}/{max_attempts} in {delay}s")
            time.sleep(delay)
        except Exception:
            raise                     # any other error: bubble up
    raise RuntimeError(f"Whisper failed after {max_attempts} attempts")

# -------- Silence Removal --------
def remove_silence(path: str) -> AudioSegment:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    filt = (
        "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:"
        "stop_periods=-1:stop_duration=0.3:stop_threshold=-50dB"
    )
    with Spinner(desc="[STEP] Removing silence…"):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", path,
            "-af", filt,
            out_path
        ], check=True)
    cleaned = AudioSegment.from_file(out_path)
    print_and_flush(f"[OK] Silence-removed MP3 ready: {out_path}")
    os.unlink(out_path)
    return cleaned


# -------- Chunking for Whisper --------
def chunk_audio(audio: AudioSegment) -> List[str]:
    total_ms = len(audio)
    limit_ms = SECONDS_LIMIT * 1000
    paths = []
    start = 0
    idx = 1
    chunks_estimate = max(1, int(total_ms // limit_ms) + 1)
    print_and_flush(f"[STEP] Chunking audio: {total_ms/1000:.1f} seconds total, {SECONDS_LIMIT} s per chunk max")
    with tqdm(total=chunks_estimate, desc="[STEP] Chunking", unit="chunk") as pbar:
        while start < total_ms:
            end = min(start + limit_ms, total_ms)
            # backtrack to nearest quiet spot up to 2s
            if end < total_ms:
                snippet = audio[end-2000:end]
                if snippet.dBFS < -45:
                    end -= 2000
            seg = audio[start:end]
            data = seg_to_bytes(seg)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            with open(tmp, "wb") as f:
                f.write(data)
            size_mb = len(data)/1_048_576
            dur_s = len(seg)/1000
            print_and_flush(f"[INFO]   Chunk {idx}: {dur_s:.1f}s, {size_mb:.2f} MB")
            paths.append(tmp)
            idx += 1
            start = end
            pbar.update(1)
    print_and_flush(f"[OK] Chunking complete. {len(paths)} chunk(s) ready.")
    return paths


# -------- Parallel Transcription --------
def transcribe_paths(
    client: openai.OpenAI,
    paths: List[str],
    model: str = "whisper-1",
) -> List[str]:
    """
    Transcribe already-chunked paths concurrently.
    Falls back to original ordering.
    """
    texts = [None] * len(paths)

    def worker(i_p):
        i, path = i_p
        txt = _whisper_with_retry(client, path, model)
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

# -------- Diarization & Speaker Identification --------
def diarize_and_transcribe(
    cleaned_audio: AudioSegment,
    output_txt: str,
    client: openai.OpenAI,
    model: str = "whisper-1",
):
    """
    Run speaker diarization, split any over-large segments, send requests
    concurrently, then re-assemble the transcript in chronological order.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print_and_flush("[FATAL] pyannote.audio is required for diarization. Install with 'pip install pyannote.audio'")
        raise

    # --- 1. Diarize ---------------------------------------------------------
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

    # --- 2. Build transcription jobs ---------------------------------------
    limit_ms = SECONDS_LIMIT * 1000
    jobs = []                       # (global_order, speaker_name, AudioSegment)
    speaker_map, speaker_counter = {}, 1
    order_counter = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg_len = turn.end - turn.start
        if seg_len < 0.11:          # Whisper hard minimum 0.1 s
            continue

        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"; speaker_counter += 1
        spk_name = speaker_map[speaker]

        seg = audio[int(turn.start * 1000):int(turn.end * 1000)]

        # split if segment exceeds Whisper’s byte/second ceiling
        parts = [seg[i:i + limit_ms] for i in range(0, len(seg), limit_ms)]
        for part in parts:
            jobs.append((order_counter, spk_name, part))
            order_counter += 1

    # --- 3. Concurrent Whisper calls ---------------------------------------
    results = [None] * len(jobs)

    def worker(job):
        idx, spk, seg = job
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        seg.export(tmp, format="mp3")
        try:
            text = _whisper_with_retry(client, tmp, model)
        finally:
            os.unlink(tmp)
        return idx, spk, text

    with tqdm(total=len(jobs), desc="[STEP] Transcribing diarized segments", unit="segment") as pbar, \
         ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as exe:

        for fut in as_completed([exe.submit(worker, j) for j in jobs]):
            idx, spk, txt = fut.result()
            results[idx] = (spk, txt)
            pbar.update(1)

    # --- 4. Re-assemble & write --------------------------------------------
    lines = [f"[{spk}] {txt}" for spk, txt in results if txt]
    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print_and_flush(f"[SUCCESS] Written diarized transcription to: {os.path.abspath(output_txt)}")
    os.unlink(wav_path)
    return os.path.abspath(output_txt)


# -------- Main Workflow --------
def transcribe_file(
    input_path: str,
    output_txt: str = "transcription.txt",
    diarize: bool = False
) -> str:
    print_and_flush(f"[ALIVE] Script launched. Transcribing: {input_path}")

    # --- OpenAI client with sane defaults & retries ------------------------
    if OPENAI_API_KEY == "your-api-key-here":
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")

    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=httpx.Timeout(120.0, read=PER_REQUEST_TIMEOUT_S),
        max_retries=3,
    )
    print_and_flush("[OK] OpenAI client initialized.")

    # --- unchanged work: convert, silence-remove ---------------------------
    input_path = convert_to_mp3(input_path)
    cleaned = remove_silence(input_path)

    if diarize:
        if not HUGGINGFACE_TOKEN:
            raise RuntimeError("Set HUGGINGFACE_TOKEN to accept model licenses.")
        return diarize_and_transcribe(cleaned, output_txt, client)
    else:
        paths = chunk_audio(cleaned)
        texts = transcribe_paths(client, paths)
        with open(output_txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(texts))
        print_and_flush(f"[SUCCESS] Finished! Transcription written to: {os.path.abspath(output_txt)}")
        return os.path.abspath(output_txt)


# --- Entry Point ---
if __name__ == "__main__":
    print_and_flush(f"[BOOT] Script running as __main__ with args: {sys.argv}")
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker diarization.")
    parser.add_argument("input_file", help="Input audio file (wav/mp3/etc.)")
    parser.add_argument("output_file", nargs="?", default="transcription.txt", help="Output text file")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker identification/diarization (requires pyannote.audio and HuggingFace token)")
    args = parser.parse_args()

    try:
        transcribe_file(args.input_file, args.output_file, diarize=args.diarize)
    except Exception as e:
        print_and_flush(f"[FATAL ERROR] Script exited with error: {e}")
        exit(1)
