import asyncio
import json
import logging
import os
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import re

try:  # pragma: no cover - optional dependency
    from vosk import KaldiRecognizer as KaldiRecognizerRuntime
    from vosk import Model as VoskModelRuntime
except Exception:  # pragma: no cover - optional dependency
    KaldiRecognizerRuntime = None  # type: ignore[assignment]
    VoskModelRuntime = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from vosk import KaldiRecognizer as VoskRecognizerType
    from vosk import Model as VoskModelType
else:  # pragma: no cover - fallback types for runtime when vosk missing
    VoskRecognizerType = Any
    VoskModelType = Any

import webapp

logger = logging.getLogger(__name__)
if os.environ.get("STREAMING_DEBUG"):
    logger.setLevel(logging.DEBUG)

STREAM_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2  # int16 bytes
MIN_PARTIAL_INTERVAL = 0.35  # seconds between partial pushes


_VOSK_MODEL: Optional["VoskModelType"] = None


TS_RANGE_PATTERN = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\-(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*(?P<rest>.*)$"
)
SPEAKER_PATTERN = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<rest>.*)$")


def _parse_hhmmss(value: str) -> Optional[float]:
    try:
        hh, mm, ss = value.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return None


def extract_segments(transcript: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not transcript:
        return segments
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        start: Optional[float] = None
        end: Optional[float] = None
        speaker: Optional[str] = None
        rest = line
        m_range = TS_RANGE_PATTERN.match(rest)
        if m_range:
            start = _parse_hhmmss(m_range.group("start"))
            end = _parse_hhmmss(m_range.group("end"))
            rest = m_range.group("rest").lstrip()
        m_speaker = SPEAKER_PATTERN.match(rest)
        if m_speaker:
            speaker = m_speaker.group("speaker").strip()
            rest = m_speaker.group("rest").strip()
        if not rest:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": rest,
            }
        )
    return segments


def get_vosk_model() -> Optional["VoskModelType"]:
    global _VOSK_MODEL
    if _VOSK_MODEL is not None:
        return _VOSK_MODEL

    if VoskModelRuntime is None:
        logger.warning("Vosk model not available; realtime partials disabled.")
        return None

    model_dir = os.environ.get("VOSK_MODEL_DIR")
    if not model_dir:
        # Default to models/vosk if present
        candidate = os.path.join(os.path.dirname(__file__), "..", "models", "vosk")
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            model_dir = candidate
    if not model_dir or not os.path.exists(model_dir):
        logger.warning("Vosk model directory missing. Set VOSK_MODEL_DIR to enable realtime partials.")
        return None

    logger.info("Loading Vosk model from %s", model_dir)
    _VOSK_MODEL = VoskModelRuntime(model_dir)
    return _VOSK_MODEL


@dataclass
class StreamingSession:
    ws: WebSocket
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    language: Optional[str] = None
    mode: Optional[str] = "api"
    local_model: str = "base"
    diarize: bool = True
    timestamps: bool = False
    sample_rate: int = STREAM_SAMPLE_RATE
    pcm_buffer: bytearray = field(default_factory=bytearray)
    consumer_task: Optional[asyncio.Task] = None
    stopped: bool = False
    recognizer: Optional["VoskRecognizerType"] = None
    last_partial_emit: float = 0.0
    partial_text: str = ""
    activity_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    job_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    stream_position: float = 0.0
    partial_history: List[Dict[str, Any]] = field(default_factory=list)
    ws_closed: bool = False

    def configure(self, payload: Dict[str, Any]) -> None:
        self.language = payload.get("language") or None
        self.mode = payload.get("mode") or "api"
        self.local_model = payload.get("local_model") or "base"
        self.diarize = bool(payload.get("diarize", True))
        self.timestamps = bool(payload.get("timestamps", False))
        self.started_at = time.time()
        self.stream_position = 0.0
        self.partial_history.clear()
        self.partial_text = ""
        self.last_partial_emit = 0.0
        sample_rate = payload.get("sample_rate")
        try:
            if sample_rate:
                self.sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            self.sample_rate = STREAM_SAMPLE_RATE

        vosk_model = get_vosk_model()
        if vosk_model is not None and KaldiRecognizerRuntime is not None:
            try:
                recognizer = KaldiRecognizerRuntime(vosk_model, self.sample_rate)
                recognizer.SetWords(True)
                self.recognizer = recognizer
                logger.info("Vosk recognizer initialised for session %s", self.id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialise Vosk recognizer: %s", exc)
                self.recognizer = None
        else:
            self.recognizer = None

    async def handle_audio_chunk(self, chunk: bytes, ts: float) -> None:
        if not chunk:
            return
        _ = ts  # wall-clock timestamp currently unused; keep for future diagnostics
        samples = len(chunk) // PCM_SAMPLE_WIDTH
        if samples <= 0:
            return
        duration = samples / float(self.sample_rate or STREAM_SAMPLE_RATE)
        self.stream_position += duration
        position = self.stream_position
        self.pcm_buffer.extend(chunk)
        if int(self.stream_position * 10) % 10 == 0:  # log roughly each second
            logger.debug(
                "Session %s received audio chunk: %d samples (%.2fs total)",
                self.id,
                samples,
                self.stream_position,
            )
        if not self.recognizer:
            return

        try:
            accepted = self.recognizer.AcceptWaveform(chunk)
            if accepted:
                result = json.loads(self.recognizer.Result())
                text = (result or {}).get("text")
                if text:
                    await self.emit_partial(text, final=True, ts=position)
                    self.partial_text = ""
            else:
                partial_raw = self.recognizer.PartialResult()
                payload = json.loads(partial_raw)
                text = (payload or {}).get("partial")
                if text and len(text.strip()) >= 1:
                    if position - self.last_partial_emit >= MIN_PARTIAL_INTERVAL:
                        await self.emit_partial(text, final=False, ts=position)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Recognizer failure: %s", exc)
            self.recognizer = None

    async def emit_partial(self, text: str, final: bool, ts: float) -> None:
        if self.ws_closed:
            return
        async with self.activity_lock:
            try:
                await self.ws.send_json(
                    {
                        "type": "partial",
                        "text": text,
                        "final": final,
                        "timestamp": ts,
                    }
                )
            except Exception as exc:
                logger.debug("Failed to send partial for session %s: %s", self.id, exc)
                self.ws_closed = True
            else:
                if final:
                    self.partial_history.append({"text": text, "time": ts})
                    if len(self.partial_history) > 500:
                        del self.partial_history[:-500]
                self.last_partial_emit = ts

    async def send_event(self, payload: Dict[str, Any]) -> None:
        if self.ws_closed:
            return
        async with self.activity_lock:
            if self.ws_closed:
                return
            try:
                await self.ws.send_json(payload)
            except (RuntimeError, WebSocketDisconnect) as exc:
                self.ws_closed = True
                logger.debug("Stream %s send failed: %s", self.id, exc)
            except Exception as exc:  # pragma: no cover - defensive
                self.ws_closed = True
                logger.debug("Stream %s send failed (unexpected): %s", self.id, exc)

    async def finalise(self) -> None:
        import logging
        logger = logging.getLogger("streaming.server.finalise")
        logger.debug(f"[finalise] Called for session {self.id}, stopped={self.stopped}, pcm_buffer={len(self.pcm_buffer)} bytes")
        if self.stopped:
            logger.debug(f"[finalise] Already stopped for session {self.id}")
            return
        self.stopped = True
        await self.send_event({"type": "status", "message": "Finalising recording"})
        if not self.pcm_buffer:
            logger.debug(f"[finalise] No audio captured for session {self.id}")
            await self.send_event({"type": "error", "message": "No audio captured"})
            return
        loop = asyncio.get_running_loop()
        logger.debug(f"[finalise] Launching final job for session {self.id}")
        job_info = await loop.run_in_executor(None, self._launch_final_job)
        if job_info is None:
            logger.debug(f"[finalise] Failed to schedule transcription job for session {self.id}")
            await self.send_event({"type": "error", "message": "Failed to schedule transcription job"})
            return
        self.job_id = job_info["job_id"]
        await self.send_event(
            {
                "type": "status",
                "message": "Running high-accuracy transcription",
                "job_id": self.job_id,
            }
        )
        logger.debug(f"[finalise] Starting _watch_job for session {self.id}, job_id={self.job_id}")
        asyncio.create_task(self._watch_job(job_info))

    def _launch_final_job(self) -> Optional[Dict[str, Any]]:
        jid = uuid.uuid4().hex[:12]
        in_path = os.path.join(webapp.UPLOAD_DIR, f"{jid}_stream.wav")
        out_path = os.path.join(webapp.OUT_DIR, f"{jid}.txt")
        os.makedirs(os.path.dirname(in_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with wave.open(in_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(PCM_SAMPLE_WIDTH)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(self.pcm_buffer))

        job = webapp.Job(id=jid, infile=in_path, outfile=out_path)
        job.options.update(
            {
                "mode": self.mode,
                "language": self.language,
                "local_model": self.local_model,
                "diarize": self.diarize,
                "aggregate": True,
                "timestamps": self.timestamps,
                "trim_silence": False,
            }
        )
        job.tracker = webapp.JobProgressTracker(diarize=self.diarize, mode=self.mode)
        job.progress_meta = job.tracker.snapshot()
        job.stage = job.progress_meta.get("stage_label", job.stage)
        job.stage_detail = job.progress_meta.get("description", job.stage_detail)
        job.progress = job.progress_meta.get("overall", job.progress)

        with webapp.jobs_lock:
            webapp.jobs[jid] = job

        def _run_job() -> None:
            webapp._run_job(  # type: ignore[attr-defined]
                job,
                mode=self.mode,
                local_model=self.local_model,
                language=self.language,
                diarize=self.diarize,
                aggregate=True,
                timestamps=self.timestamps,
                mlx_batch_size=12,
                mlx_quant=None,
                trim_silence=False,
            )

        worker = threading.Thread(target=_run_job, name=f"stream-job-{jid}", daemon=True)
        worker.start()

        return {
            "job": job,
            "job_id": jid,
            "result_path": out_path,
            "thread": worker,
        }

    async def _watch_job(self, job_info: Dict[str, Any]) -> None:
        import logging
        logger = logging.getLogger("streaming.server._watch_job")
        job: webapp.Job = job_info["job"]
        worker: threading.Thread = job_info["thread"]
        logger.debug(f"[_watch_job] Start watching job {job.id} (status={job.status})")
        try:
            while True:
                status_payload = {
                    "type": "final_progress",
                    "job_id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "stage": job.stage,
                    "description": job.stage_detail,
                }
                logger.debug(f"[_watch_job] Sending final_progress for job {job.id}: status={job.status}, progress={job.progress}")
                await self.send_event(status_payload)
                if job.status in {"done", "error"}:
                    logger.debug(f"[_watch_job] Job {job.id} finished with status={job.status}")
                    break
                await asyncio.sleep(0.5)
            loop = asyncio.get_running_loop()
            logger.debug(f"[_watch_job] Waiting for worker thread to join for job {job.id}")
            await loop.run_in_executor(None, worker.join)
        except Exception as exc:
            logger.exception(f"[_watch_job] Watcher error for job {job.id}: {exc}")
        finally:
            if job.status == "done":
                logger.debug(f"[_watch_job] Sending final for job {job.id}")
                segments = extract_segments(job.result or "")
                await self.send_event(
                    {
                        "type": "final",
                        "job_id": job.id,
                        "download_url": f"/download/{job.id}",
                        "transcript": job.result or "",
                        "segments": segments,
                        "partials": list(self.partial_history),
                    }
                )
            else:
                logger.debug(f"[_watch_job] Sending error for job {job.id}: {job.error}")
                await self.send_event(
                    {
                        "type": "error",
                        "message": job.error or "Live transcription failed",
                    }
                )


def create_streaming_app() -> FastAPI:
    streaming_app = FastAPI(title="WhisperTTS Streaming Service")
    streaming_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sessions: Dict[str, StreamingSession] = {}

    @streaming_app.get("/health")
    async def healthcheck():
        return JSONResponse({"status": "ok"})

    @streaming_app.websocket("/ws/stream")
    async def websocket_stream(ws: WebSocket):
        await ws.accept()
        session = StreamingSession(ws=ws)
        sessions[session.id] = session
        await session.send_event({"type": "ready", "session_id": session.id})
        try:
            while True:
                message = await ws.receive()
                if "type" in message and message["type"] == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    chunk = message["bytes"]
                    timestamp = time.time()
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Chunk received for session %s: %d bytes", session.id, len(chunk))
                    await session.handle_audio_chunk(chunk, ts=float(timestamp))
                    continue
                text = message.get("text")
                if text:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        await session.send_event({"type": "error", "message": "Invalid JSON"})
                        continue
                    msg_type = payload.get("type")
                    logger.debug("Text message for session %s: %s", session.id, msg_type)
                    if msg_type == "start":
                        session.configure(payload)
                        await session.send_event({"type": "status", "message": "Streaming started"})
                    elif msg_type == "stop":
                        await session.finalise()
                        break
                    elif msg_type == "ping":
                        await session.send_event({"type": "pong", "time": payload.get("time")})
                    else:
                        await session.send_event({"type": "error", "message": f"Unknown message type: {msg_type}"})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session %s", session.id)
        except Exception as exc:
            logger.exception("WebSocket error: %s", exc)
            try:
                await session.send_event({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            sessions.pop(session.id, None)
            if not session.stopped and session.pcm_buffer:
                await session.finalise()

    return streaming_app


streaming_app = create_streaming_app()
