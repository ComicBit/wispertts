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

---

## License

MIT © Leandro Piccione


## Web Interface

A simple Flask web server is provided to run the transcription through a browser.

### Usage

```bash
pip install -r requirements.txt
python webapp.py
```

Open `http://localhost:5000` and upload an audio file. Select desired options,
start the transcription and follow the progress. Once finished, the transcript
can be viewed in the browser or downloaded as a text file.
