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
from tqdm import tqdm

# -------- Configuration --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Whisper API file-size limit
MAX_WHISPER_MB = 25
SAFETY_MARGIN = 0.95
TARGET_BYTES = int(MAX_WHISPER_MB * 1_048_576 * SAFETY_MARGIN)
BITRATE_KBPS = 128
SECONDS_LIMIT = int(TARGET_BYTES / (BITRATE_KBPS/8*1024))

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
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", path,
        "-af", filt,
        out_path
    ], check=True)
    cleaned = AudioSegment.from_file(out_path)
    os.unlink(out_path)
    return cleaned

# -------- Chunking for Whisper --------
def chunk_audio(audio: AudioSegment) -> List[str]:
    total_ms = len(audio)
    limit_ms = SECONDS_LIMIT * 1000
    paths = []
    start = 0
    idx = 1
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
        print(f"  Chunk {idx}: {dur_s:.1f}s, {size_mb:.2f} MB")
        paths.append(tmp)
        idx += 1
        start = end
    return paths

# -------- Parallel Transcription --------
def transcribe_paths(client: openai.OpenAI, paths: List[str], model: str = "whisper-1") -> List[str]:
    texts = [None] * len(paths)
    def worker(i_p):
        i, p = i_p
        with open(p, "rb") as f:
            txt = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="text"
            )
        return i, txt

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(worker, (i, p)): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Transcribing", unit="chunk"):
            i, t = fut.result()
            texts[i] = t
    return texts

# -------- Main Workflow --------
def transcribe_file(
    input_path: str,
    output_txt: str = "transcription.txt",
    diarize: bool = False
) -> str:
    if OPENAI_API_KEY == "your-api-key-here":
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # 1) Silence removal
    print("Removing silence…")
    cleaned = remove_silence(input_path)

    if diarize:
        # 2) Speaker diarization in chunks with progress
        if not HUGGINGFACE_TOKEN:
            raise RuntimeError("Set HUGGINGFACE_TOKEN to accept model licenses.")
        from pyannote.audio import Pipeline

        # choose device: prioritize CUDA, then MPS (Apple Silicon), else CPU
        # select compute device: MPS (Apple), CUDA (NVIDIA), or CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"Initializing diarization on {device}…")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        pipeline.to(device)
