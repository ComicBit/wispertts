# Audio Transcription Script with Silence Removal & Chunking

This script automates the process of transcribing long audio files using OpenAI Whisper API, while optimizing for speed and cost by:

- **Removing silence** using FFmpeg’s high-performance `silenceremove` filter.
- **Splitting** cleaned audio into chunks that stay under Whisper’s 25 MB limit (with safety margin).  
- **Transcribing** chunks in parallel to speed up upload and API response.  
- **Merging** all partial transcripts into a single text output.

---

## Features

- **Fast silence trimming** via FFmpeg (no slow Python loops).  
- **Automatic chunking** based on file-size constraints and audio bitrate.  
- **Parallel uploads** with progress bars (using `tqdm` & `concurrent.futures`).  
- **Single unified transcript** at the end.  
- **Configurable** thresholds, bitrate assumptions, FFmpeg filter parameters, and worker count.

---

## Prerequisites

1. **Python 3.7+** installed.  
2. **FFmpeg** installed and on your PATH:
   - macOS: `brew install ffmpeg`
   - Debian/Ubuntu: `sudo apt install ffmpeg`
   - Windows: download from https://ffmpeg.org
3. **Python dependencies** (install via pip):
   ```bash
   pip install openai pydub tqdm
   ```
4. **OpenAI API Key** with Whisper access.  
   - Set as environment variable:
     ```bash
     export OPENAI_API_KEY="sk-..."
     ```

---

## Installation

1. Clone or download this repository/script.  
2. Ensure `script.py` (the transcription script) is executable or run with Python.  
3. Verify you can invoke `ffmpeg` from your shell.

---

## Configuration

All key parameters live near the top of `script.py`:

- `MAX_WHISPER_MB` and `SAFETY_MARGIN` – define the 25 MB API limit and buffer.  
- `BITRATE_KBPS` – assumed mp3 bitrate (default 128 kbps).  
- FFmpeg `silenceremove` filter options:
  - `start_duration`, `stop_duration` (in seconds) – minimum silence length to trim.  
  - `threshold` (e.g. `-50dB`) – volume threshold for silence.  
- `workers` – number of parallel API uploads.

Feel free to tweak these for different file types or quality targets.

---

## Usage

Run from the command line:

```bash
python script.py <input_audio> -o <output_text>
```

- `<input_audio>`: path to your source file (mp3, wav, m4a, etc.).  
- `-o <output_text>`: (optional) path for the final transcript text file. Defaults to `transcription.txt`.

**Example:**
```bash
python script.py lecture_recording.mp3 -o lecture.txt
```

The script will log:
1. **Silence removal** start, output path, and cleaned duration.  
2. **Chunking** details (duration and size of each piece).  
3. **Transcription** progress bar for parallel uploads.  
4. **Final** transcript save location.

---

## How It Works

1. **Silence Removal**  
   Uses FFmpeg’s `silenceremove` to drop silent segments at the start and within the audio. This is extremely fast (C-based).  

2. **Chunking**  
   - Computes a maximum time window per chunk based on `TARGET_BYTES` and `BITRATE_KBPS`.  
   - Slices the cleaned audio into contiguous segments ≤ limit.  
   - Attempts a small backtrack (~2 s) to avoid cutting mid-speech if the end is noisy.  
   - Exports each chunk exactly once to MP3, naming them in a temp directory.  

3. **Parallel Transcription**  
   - Uses `concurrent.futures.ThreadPoolExecutor` to upload & transcribe multiple chunks concurrently.  
   - Shows a `tqdm` progress bar for chunk uploads and completions.  

4. **Merge & Cleanup**  
   - Gathers individual transcripts in order.  
   - Writes a single text file (`-o` argument).  
   - Deletes temporary chunk files.

---

## Tuning & Troubleshooting

- **Too much speech cut?** Increase `stop_duration` or lower `threshold` in the FFmpeg filter.  
- **Chunks still too big?** Reduce `BITRATE_KBPS` or adjust the safety margin.  
- **Too slow?** Increase `workers` for more parallel uploads (beware API rate limits).  
- **FFmpeg errors**: Run `ffmpeg -version` to confirm installation.

---

## License

MIT © Leandro Piccione

