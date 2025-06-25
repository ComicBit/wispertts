#!/usr/bin/env python3
import os
import tempfile
import subprocess
from io import BytesIO
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import torch
from pydub import AudioSegment
from tqdm import tqdm, trange
import time

# -------- Configuration --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Whisper API file-size limit
MAX_WHISPER_MB = 25
SAFETY_MARGIN = 0.95
TARGET_BYTES = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
BITRATE_KBPS = 128
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE_KBPS/8*1024))

def print_and_flush(msg):
    print(msg, flush=True)

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
def transcribe_paths(client: openai.OpenAI, paths: List[str], model: str = "whisper-1") -> List[str]:
    texts = [None] * len(paths)
    def worker(i_p):
        i, p = i_p
        try:
            with open(p, "rb") as f:
                txt = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="text"
                )
            print_and_flush(f"[OK] Transcribed chunk {i+1}/{len(paths)}")
            return i, txt
        except Exception as e:
            print_and_flush(f"[FAIL] Transcription failed for chunk {i+1}: {e}")
            raise

    with tqdm(total=len(paths), desc="[STEP] Transcribing", unit="chunk") as pbar:
        with ThreadPoolExecutor() as exe:
            futures = {exe.submit(worker, (i, p)): i for i, p in enumerate(paths)}
            for fut in as_completed(futures):
                try:
                    i, t = fut.result()
                    texts[i] = t
                except Exception as e:
                    print_and_flush(f"[FAIL] Error in thread transcription: {e}")
                    raise
                pbar.update(1)
    print_and_flush(f"[OK] All chunks transcribed.")
    return texts

# -------- Main Workflow --------
def transcribe_file(
    input_path: str,
    output_txt: str = "transcription.txt",
    diarize: bool = False
) -> str:
    print_and_flush(f"[ALIVE] Script launched. Transcribing: {input_path}")

    try:
        if OPENAI_API_KEY == "your-api-key-here":
            print_and_flush("[ERROR] OPENAI_API_KEY not set. Aborting.")
            raise RuntimeError("Set OPENAI_API_KEY environment variable.")

        print_and_flush("[STEP] Initializing OpenAI client …")
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print_and_flush("[OK] OpenAI client initialized.")
    except Exception as e:
        print_and_flush(f"[FATAL] Could not initialize OpenAI client: {e}")
        raise

    # Convert input to MP3
    print_and_flush("[STEP] Converting input to MP3 …")
    try:
        input_path = convert_to_mp3(input_path)
    except Exception as e:
        print_and_flush(f"[FATAL] Failed to convert input file to MP3: {e}")
        raise

    # Silence removal
    print_and_flush("[STEP] Removing silence …")
    try:
        cleaned = remove_silence(input_path)
    except Exception as e:
        print_and_flush(f"[FATAL] Failed during silence removal: {e}")
        raise

    if diarize:
        print_and_flush("[STEP] Diarization is enabled.")
        if not HUGGINGFACE_TOKEN:
            print_and_flush("[ERROR] HUGGINGFACE_TOKEN not set. Aborting.")
            raise RuntimeError("Set HUGGINGFACE_TOKEN to accept model licenses.")
        try:
            from pyannote.audio import Pipeline
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            print_and_flush(f"[STEP] Initializing diarization on {device} …")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HUGGINGFACE_TOKEN
            )
            pipeline.to(device)
        except Exception as e:
            print_and_flush(f"[FATAL] Diarization pipeline error: {e}")
            raise
        # [Add diarization logic here if needed]
        print_and_flush("[TODO] Diarization logic not implemented in this script.")
        raise NotImplementedError("Diarization logic not yet implemented.")

    else:
        print_and_flush("[STEP] Chunking audio …")
        try:
            paths = chunk_audio(cleaned)
        except Exception as e:
            print_and_flush(f"[FATAL] Error during chunking: {e}")
            raise

        print_and_flush("[STEP] Sending chunks to Whisper API …")
        try:
            texts = transcribe_paths(client, paths)
        except Exception as e:
            print_and_flush(f"[FATAL] Error in Whisper transcription: {e}")
            raise

        print_and_flush(f"[STEP] Writing transcription to file: {output_txt} …")
        try:
            final_text = "\n".join(texts)
            with open(output_txt, "w", encoding="utf-8") as fh:
                fh.write(final_text)
            print_and_flush(f"[SUCCESS] Finished! Transcription written to: {os.path.abspath(output_txt)}")
            return os.path.abspath(output_txt)
        except Exception as e:
            print_and_flush(f"[FATAL] Could not write output file: {e}")
            raise

# --- Entry Point ---
if __name__ == "__main__":
    import sys
    print_and_flush(f"[BOOT] Script running as __main__ with args: {sys.argv}")
    if len(sys.argv) < 2:
        print_and_flush(f"[USAGE] {sys.argv[0]} INPUT_FILE [OUTPUT_FILE]")
        exit(1)
    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "transcription.txt"
    try:
        transcribe_file(in_file, out_file)
    except Exception as e:
        print_and_flush(f"[FATAL ERROR] Script exited with error: {e}")
        exit(1)
