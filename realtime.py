#!/usr/bin/env python3
"""Realtime microphone transcription using Whisper.

This module records audio from the local microphone in short chunks and
transcribes them using either the OpenAI API or a local Whisper model.  It
mirrors the CLI structure of ``script.py`` while printing live transcripts to
STDOUT.  An optional ``--web`` flag launches a tiny FastAPI server which allows
streaming microphone audio from the browser via WebSocket.  All recognised
speech is appended to the chosen output text file.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
from typing import List, Optional

import sounddevice as sd
from pydub import AudioSegment

# Optional web dependencies
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except Exception:  # pragma: no cover - only needed for --web
    FastAPI = None  # type: ignore

import openai

# Reuse helpers from the main script
import script


CHUNK_SECONDS = 5  # duration of microphone chunks


def record_chunks(duration: int = CHUNK_SECONDS):
    """Yield :class:`pydub.AudioSegment` chunks from the microphone."""

    samplerate = 16_000
    channels = 1
    frames = int(duration * samplerate)
    sd.default.samplerate = samplerate
    sd.default.channels = channels

    print("[BOOT] Listening – press Ctrl+C to stop", flush=True)
    try:
        while True:
            data = sd.rec(frames, dtype="int16")
            sd.wait()
            seg = AudioSegment(
                data.tobytes(),
                frame_rate=samplerate,
                sample_width=data.dtype.itemsize,
                channels=channels,
            )
            yield seg
    except KeyboardInterrupt:
        return


def transcribe_segment(seg: AudioSegment, *, mode: str, cli, mdl, diarize: bool, lang: Optional[str]):
    if diarize:
        rows = script.diarize_tx(seg, mode=mode, cli=cli, mdl=mdl, lang=lang)
    else:
        rows = script.transcribe_chunks(script.chunk(seg), mode=mode, cli=cli, mdl=mdl, lang=lang)
    return rows


def run_local(outfile: str, *, mode: str, local_tag: str, diarize: bool, lang: Optional[str], show_ts: bool, agg: bool):
    cli = mdl = None
    if mode == "api":
        if script.OPENAI_API_KEY == "your-api-key-here":
            raise RuntimeError("Set OPENAI_API_KEY")
        cli = openai.OpenAI(api_key=script.OPENAI_API_KEY)
    else:
        mdl = script.load_local(local_tag, script.prefer_device())

    all_rows: List = []
    for seg in record_chunks():
        if seg is None:
            break
        rows = transcribe_segment(seg, mode=mode, cli=cli, mdl=mdl, diarize=diarize, lang=lang)
        if agg:
            rows = script.merge(rows)
        for line in script.to_lines(rows, show_ts):
            print(line)
        all_rows.extend(rows)

    if agg:
        all_rows = script.merge(all_rows)
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(script.to_lines(all_rows, show_ts)))
    print(f"[SUCCESS] Transcript → {os.path.abspath(outfile)}")


# --------------------------- WebSocket server ---------------------------
CLIENT_HTML = """
<!doctype html>
<html>
<body>
<button id=\"start\">Start</button>
<button id=\"stop\">Stop</button>
<pre id=\"log\"></pre>
<script>
let ws, rec;
const log = document.getElementById('log');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
start.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  rec = new MediaRecorder(stream);
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => { log.textContent += e.data + '\n'; };
  rec.ondataavailable = e => ws.send(e.data);
  rec.start(1000);
};
stop.onclick = () => { rec.stop(); ws.close(); };
</script>
</body>
</html>
"""


def run_server(outfile: str, *, mode: str, local_tag: str, diarize: bool, lang: Optional[str], show_ts: bool, agg: bool):
    if FastAPI is None:
        raise RuntimeError("fastapi and uvicorn are required for --web mode")

    cli = mdl = None
    if mode == "api":
        if script.OPENAI_API_KEY == "your-api-key-here":
            raise RuntimeError("Set OPENAI_API_KEY")
        cli = openai.OpenAI(api_key=script.OPENAI_API_KEY)
    else:
        mdl = script.load_local(local_tag, script.prefer_device())

    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(CLIENT_HTML)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        rows: List = []
        try:
            while True:
                data = await ws.receive_bytes()
                seg = AudioSegment.from_file(io.BytesIO(data), format="webm")
                seg_rows = transcribe_segment(seg, mode=mode, cli=cli, mdl=mdl, diarize=diarize, lang=lang)
                if agg:
                    seg_rows = script.merge(seg_rows)
                for line in script.to_lines(seg_rows, show_ts):
                    print(line)
                    await ws.send_text(line)
                rows.extend(seg_rows)
        except WebSocketDisconnect:
            pass
        if agg:
            rows = script.merge(rows)
        with open(outfile, "a", encoding="utf-8") as fh:
            fh.write("\n".join(script.to_lines(rows, show_ts)) + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


# ------------------------------ CLI -----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Realtime microphone transcription")
    ap.add_argument("output_file")
    ap.add_argument("--mode", choices=["api", "local"], default="api")
    ap.add_argument("--local-model", default="base")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--language", default=None, metavar="ISO")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--web", action="store_true", help="serve browser client instead of local mic")
    args = ap.parse_args()

    if args.web:
        run_server(
            args.output_file,
            mode=args.mode,
            local_tag=args.local_model,
            diarize=args.diarize,
            lang=args.language,
            show_ts=args.timestamps,
            agg=args.aggregate,
        )
    else:
        run_local(
            args.output_file,
            mode=args.mode,
            local_tag=args.local_model,
            diarize=args.diarize,
            lang=args.language,
            show_ts=args.timestamps,
            agg=args.aggregate,
        )


if __name__ == "__main__":
    main()
