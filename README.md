# Audio → Text CLI (OpenAI Whisper + pyannote)

This script automates **accurate** & **fast** transcription of long recordings while keeping costs predictable.

- **Silence trimming** with FFmpeg (`silenceremove`).
- **Smart chunking** under Whisper’s 25 MB / 15 min limit.
- **Parallel uploads** with built‑in *timeouts* & *auto‑retries*.
- **(Opt‑in) Speaker diarization** via **pyannote.audio**.
- **Local **or** OpenAI‑API** inference: device picks CUDA ► MPS ► CPU automatically.
- **Optional timestamps** and **line aggregation** for cleaner output.

---

## Key Features

| Area                         | What you get                                                   |
| ---------------------------- | -------------------------------------------------------------- |
| Silence Removal              | C‑speed trimming before upload – saves money & time           |
| Chunking                     | Cuts to size, avoids mid‑speech splits, safety margin applied |
| Concurrency                  | `ThreadPoolExecutor` + progress bars, adjustable pool         |
| Retries & Rate‑Limits        | Exponential back‑off on `RateLimitError`/timeouts             |
| Diarization (`--diarize`)    | pyannote 3.1 pipeline with live progress bars                 |
| Language (`--language de`)   | Force Whisper language instead of auto‑detect                 |
| Timestamps (`--timestamps`)  | `[HH:MM:SS‑HH:MM:SS]` per line                                |
| Aggregation (`--aggregate`)  | Merge consecutive lines from the same speaker                 |
| Device Selection             | CUDA ► MPS ► CPU – no flags needed                             |
| Local Whisper (`--mode local`)| Run tiny–large‑v3 offline, weight cache in `models/whisper/`  |

---

## Quick Start

```bash
pip install "openai>=1.0" openai-whisper pydub tqdm httpx torch
# Optional diarization dependencies
pip install pyannote.audio soundfile

# macOS FFmpeg (Linux: apt, Windows: choco/scoop)
brew install ffmpeg

export OPENAI_API_KEY="sk-..."
# For diarization only ↓
export HUGGINGFACE_TOKEN="hf_..."
```

### Basic transcription (OpenAI API)

```bash
python script.py meeting.wav output.txt
```

### Force Italian, with timestamps & aggregation

```bash
python script.py meeting.wav it_meeting.txt \
       --language it --timestamps --aggregate
```

### Full diarized transcript (needs HF token)

```bash
python script.py panel.mp3 panel.txt \
       --diarize --timestamps --aggregate
```

### Offline, local tiny model (CUDA/MPS/CPU)

```bash
python script.py interview.flac out.txt \
       --mode local --local-model tiny --timestamps
```

---

## CLI Arguments

| Flag                 | Default             | Description                                             |
| -------------------- | ------------------- | ------------------------------------------------------- |
| `input_file`         |  –                  | Audio file (wav/mp3/aac/…)                              |
| `output_file`        | `transcription.txt` | Text output                                             |
| `--mode`             | `api`               | `api` = OpenAI endpoint, `local` = on‑device Whisper    |
| `--local-model TAG`  | `base`              | Whisper checkpoint (`tiny`, `small.en`, `large‑v3`, …) |
| `--diarize`          | _off_               | Identify speakers with pyannote                         |
| `--language ISO`     | auto                | Whisper language code (`en`, `it`, …)                   |
| `--timestamps`       | _off_               | Include start‑end timestamp per line                    |
| `--aggregate`        | _off_               | Merge consecutive lines from same speaker               |

---

## How It Works

1. **Convert → MP3** – normalises container for predictable size.
2. **Trim Silence** – `ffmpeg -af silenceremove=…`.
3. **Chunk** – respects 25 MB limit, backtracks ≤2 s on loud cuts.
4. **(Optional) Diarize** – pyannote on fastest device with progress hook.
5. **Transcribe** – parallel Whisper with 90 s timeout, 5× exponential retries.
6. **Post‑process** – timestamps + optional aggregation.

---

## Tuning

| Knob                    | Where                                    |
| ----------------------- | -----------------------------------------|
| Workers                 | `MAX_CONC_REQUESTS` constant (default 4) |
| Silence sensitivity     | Edit `silenceremove` line in `remove_silence()` |
| Per‑request timeout     | `PER_REQ_TIMEOUT` constant (seconds)     |

---

## Troubleshooting

| Issue                          | Fix                                                      |
| ------------------------------ | -------------------------------------------------------- |
| *CUDA not detected*            | Check `nvidia-smi`; install PyTorch + CUDA toolkit       |
| `RateLimitError` loops         | Lower workers or request higher OpenAI quota            |
| `ffmpeg not found`             | Ensure FFmpeg in PATH (`ffmpeg -version`)               |
| pyannote download/auth errors  | Export valid `HUGGINGFACE_TOKEN`; accept model licence  |
| MPS falls back to CPU          | macOS builds now auto‑use MLX (`mlx-whisper`); reinstall weights under `./mlx_models` if Apple MLX packages were removed |

---

## License

MIT © Leandro Piccione


## Web Interface

A simple Flask web server is provided to run the transcription through a browser.

### Usage

# wispertts — Audio → Text (CLI + Web)

This repository provides both:

- A CLI transcription tool (`script.py`) that uses OpenAI Whisper (API, local, or MLX) and optional speaker diarization via `pyannote.audio`.
- A small Flask web UI (`webapp.py`) to upload audio and run transcriptions from a browser.

This README shows how to install, run the CLI, and run the web UI on macOS (zsh). It includes a small helper script to automate setup and launching.

## Quick setup (recommended)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio processing):

macOS (Homebrew):
```bash
brew install ffmpeg
```

Linux (Debian/Ubuntu):
```bash
sudo apt update && sudo apt install -y ffmpeg
```

4. Export API keys (if using OpenAI or HuggingFace models):

```bash
export OPENAI_API_KEY="sk-..."
# Optional: for diarization (pyannote)
export HUGGINGFACE_TOKEN="hf_..."
```

## Run the CLI

Basic usage:

```bash
python script.py input_audio.wav output.txt
```

Common options:
- --mode: `api` (default), `local`, or `mlx` (Apple MLX)
- --local-model: local whisper tag (e.g. `tiny`, `base`, `large-v3`)
- --diarize: enable speaker diarization (requires `HUGGINGFACE_TOKEN`)
- --language: force language code (e.g. `it`, `de`)
- --timestamps / --aggregate: formatting options

See `script.py --help` for the full list of flags.

## Run the Web UI

Start the Flask server:

```bash
python webapp.py
```

Open http://localhost:5001 in your browser. Upload an audio file, choose options and start the transcription. The web UI runs transcriptions in background threads and exposes a status page and download link when complete.

## Helper installer / launcher

A convenience script is included at `scripts/run.sh`. It will:

- Create a virtual environment (if missing)
- Install Python requirements
- (Optionally) install FFmpeg on macOS via Homebrew
- Launch either the CLI or the web UI

Usage examples:

Install dependencies and start the web UI:
```bash
bash scripts/run.sh --install --web
```

Install dependencies and run a CLI transcription:
```bash
bash scripts/run.sh --install --cli -- input.wav output.txt
```

Run the web UI only (assumes deps already installed):
```bash
bash scripts/run.sh --web
```

## Troubleshooting

- If `pydub` reports "No module named 'pyaudioop'", install `pydub` into the same environment (this repo includes a small shim to help compatibility).
- Ensure `python` and `pip` point at the same environment: `which python` and `python -m pip show pydub`.
- If CUDA/MPS issues occur, check your PyTorch installation instructions on https://pytorch.org for the correct wheel for your platform.

## Contributing & License

MIT — see `LICENSE.md`.
