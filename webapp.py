#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

# Import the transcription pipeline
import script as tx


app = Flask(__name__)
# Allow large uploads (up to 512 MB)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


# Simple in-memory job store
@dataclass
class Job:
    id: str
    infile: str
    outfile: str
    status: str = "running"  # running | done | error
    log: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0
    stage: str = "Queued for processing"
    stage_detail: str = "Preparing Whisper"
    progress_meta: Dict[str, Any] = field(default_factory=dict)
    tracker: Optional[JobProgressTracker] = None
    options: Dict[str, Any] = field(default_factory=dict)


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()
run_lock = threading.Lock()


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "web_runs", "uploads")
OUT_DIR = os.path.join(os.path.dirname(__file__), "web_runs", "outputs")
_ensure_dir(UPLOAD_DIR)
_ensure_dir(OUT_DIR)


STREAMING_WS_URL_ENV = os.environ.get("STREAMING_WS_URL") or os.environ.get("STREAMING_SERVER_URL")
STREAMING_HOST_ENV = os.environ.get("STREAMING_HOST")
STREAMING_PATH_ENV = os.environ.get("STREAMING_PATH", "/ws/stream")
STREAMING_PORT_ENV = os.environ.get("STREAMING_PORT")
try:
    STREAMING_SAMPLE_RATE = int(os.environ.get("STREAMING_SAMPLE_RATE", "16000"))
except ValueError:
    STREAMING_SAMPLE_RATE = 16000


def _default_streaming_ws_url() -> str:
    if STREAMING_WS_URL_ENV:
        return STREAMING_WS_URL_ENV

    from flask import request
    from urllib.parse import urlsplit

    path = STREAMING_PATH_ENV or "/ws/stream"

    try:
        parsed_request = urlsplit(request.host_url)
    except RuntimeError:
        parsed_request = urlsplit("http://localhost/")

    host_override = STREAMING_HOST_ENV
    if host_override:
        if host_override.startswith(("ws://", "wss://")):
            base = host_override.rstrip("/")
            return base if path == "/" else f"{base}{path}"
        host = host_override
        use_port = STREAMING_PORT_ENV or "8001"
    else:
        host = parsed_request.hostname or "localhost"
        use_port = STREAMING_PORT_ENV or "8001"

    req_scheme = parsed_request.scheme or "http"
    ws_scheme = "wss" if req_scheme == "https" else "ws"

    if host.startswith(("ws://", "wss://")):
        base = host.rstrip("/")
        return base if path == "/" else f"{base}{path}"

    # If host already contains an explicit port (IPv4 or IPv6 with brackets), respect it.
    if host.startswith("[") and "]" in host:
        # IPv6 literal, possibly with port (e.g. [::1]:9000)
        closing = host.find("]")
        remainder = host[closing + 1 :]
        if remainder.startswith(":"):
            return f"{ws_scheme}://{host}{path}"
        return f"{ws_scheme}://{host}:{use_port or '8001'}{path}"

    if ":" in host:
        # Assume host already includes port (e.g. example.com:9000).
        return f"{ws_scheme}://{host}{path}"

    port = use_port or "8001"
    return f"{ws_scheme}://{host}:{port}{path}"


@app.context_processor
def inject_streaming_config():
    return {
        "streaming_config": {
            "ws_url": _default_streaming_ws_url(),
            "sample_rate": STREAMING_SAMPLE_RATE,
        }
    }


@dataclass
class StageState:
    key: str
    label: str
    weight: float
    progress: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class JobProgressTracker:
    def __init__(self, *, diarize: bool, mode: Optional[str]):
        blueprint: List[Tuple[str, str, float]] = [
            ("prepare", "Preparing transcription job", 0.04),
            ("load_model", "Loading transcription model", 0.08),
            ("convert", "Converting audio", 0.08),
            ("silence", "Removing silence", 0.08),
            ("chunk", "Splitting audio into chunks", 0.07),
            ("diarize", "Running diarization", 0.15),
            ("transcribe", "Transcribing audio", 0.45),
            ("finalize", "Finalizing transcript", 0.05),
        ]

        mode_key = (mode or "").lower()
        stages: List[StageState] = []
        for key, label, weight in blueprint:
            if key == "load_model" and mode_key == "api":
                continue
            if key == "chunk" and diarize:
                continue
            if key == "diarize" and not diarize:
                continue
            stages.append(StageState(key=key, label=label, weight=weight))

        if not stages:
            stages.append(StageState(key="prepare", label="Preparing transcription job", weight=1.0))

        total_weight = sum(stage.weight for stage in stages) or 1.0
        for stage in stages:
            stage.weight = stage.weight / total_weight

        self.stages = stages
        self.stage_map = {stage.key: stage for stage in stages}
        self.aliases = {"transcribe_segments": "transcribe"}
        self._detail_overrides: Dict[str, str] = {}

    def update_from_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = payload.get("stage")
        if not isinstance(raw, str):
            return None
        key = self.aliases.get(raw, raw)
        stage = self.stage_map.get(key)
        if stage is None:
            return None

        total = payload.get("total")
        try:
            total_val = float(total)
        except (TypeError, ValueError):
            total_val = 1.0
        if total_val <= 0:
            total_val = 1.0

        current = payload.get("current")
        try:
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = 0.0
        current_val = max(0.0, min(current_val, total_val))

        progress = current_val / total_val
        if progress > stage.progress:
            stage.progress = min(progress, 1.0)

        label = payload.get("label")
        if isinstance(label, str) and label.strip():
            stage.label = label.strip()

        extra = payload.get("extra")
        if isinstance(extra, dict):
            stage.extra = extra

        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            self._detail_overrides[key] = detail.strip()

        return self.snapshot(active_key=stage.key)

    def snapshot(self, active_key: Optional[str] = None) -> Dict[str, Any]:
        active_stage: Optional[StageState] = None
        for stage in self.stages:
            if stage.progress < 0.999:
                active_stage = stage
                break
        if active_stage is None:
            active_stage = self.stages[-1]

        if active_key:
            candidate = self.stage_map.get(active_key)
            if candidate is not None and (candidate.progress < 0.999 or active_stage.progress >= 0.999):
                active_stage = candidate

        overall = sum(stage.progress * stage.weight for stage in self.stages)
        overall = max(0.0, min(overall, 1.0))

        detail = self._detail_overrides.get(active_stage.key) or self.describe(active_stage)

        timeline: List[Dict[str, Any]] = []
        for stage in self.stages:
            if stage.progress >= 0.999:
                status = "completed"
            elif stage.key == active_stage.key:
                status = "active"
            else:
                status = "upcoming"
            timeline.append(
                {
                    "key": stage.key,
                    "label": stage.label,
                    "progress": stage.progress,
                    "status": status,
                    "extra": stage.extra,
                    "description": self._detail_overrides.get(stage.key) or self.describe(stage),
                }
            )

        return {
            "overall": overall,
            "stage_key": active_stage.key,
            "stage_label": active_stage.label,
            "stage_progress": active_stage.progress,
            "description": detail,
            "timeline": timeline,
        }

    def describe(self, stage: StageState) -> str:
        extra = stage.extra or {}
        if stage.key == "prepare":
            return "Preparing Whisper and validating settings."
        if stage.key == "load_model":
            backend = extra.get("backend")
            if backend == "mlx":
                return "Loading the MLX Whisper weights into memory."
            if backend == "local":
                device = extra.get("device", "device")
                return f"Loading a local Whisper model on {device}."
            if backend == "api":
                return "Connecting to the Whisper API."
            return "Preparing transcription model."
        if stage.key == "convert":
            return "Optimizing audio format for consistent decoding."
        if stage.key == "silence":
            return "Detecting and trimming silence to improve throughput."
        if stage.key == "chunk":
            done = extra.get("chunks_done")
            total = extra.get("chunks_total")
            if done is not None and total:
                return f"Splitting audio into {total} chunks ({done} ready)."
            return "Splitting audio into manageable chunks."
        if stage.key == "diarize":
            jobs = extra.get("jobs")
            if jobs:
                return f"Detecting speakers and preparing {jobs} segments."
            return "Detecting speaker changes in the audio."
        if stage.key == "transcribe":
            done = extra.get("units_done")
            total = extra.get("units_total")
            if done is not None and total:
                return f"Transcribing speech ({done}/{total} segments complete)."
            return "Running Whisper to extract the transcript."
        if stage.key == "finalize":
            if stage.progress >= 0.999:
                return "All done! Download the transcript below."
            return "Compiling text output and wrapping up."
        return stage.label
MODEL_LIBRARY: List[Tuple[str, str]] = [
    ("large-v3", "Large v3 • Best quality"),
    ("medium", "Medium • Balanced"),
    ("medium.en", "Medium.en • Balanced (English)"),
    ("small", "Small • Fast"),
    ("small.en", "Small.en • Fast (English)"),
    ("base", "Base • Default"),
    ("base.en", "Base.en • Default (English)"),
    ("tiny", "Tiny • Fastest"),
    ("tiny.en", "Tiny.en • Fastest (English)"),
]


def _resolve_local_backend() -> Tuple[str, str]:
    """Return (mode, friendly_label) for local execution."""

    torch_mod = getattr(tx, "torch", None)
    if torch_mod is not None and getattr(torch_mod, "cuda", None):
        try:
            if torch_mod.cuda.is_available():
                return "local", "NVIDIA GPU"
        except Exception:
            pass

    mlx_available = tx.is_apple_silicon() and bool(getattr(tx, "LightningWhisperMLX", None))
    if mlx_available:
        return "mlx", "Apple MLX"

    mps_backend = None
    try:
        mps_backend = getattr(torch_mod.backends, "mps", None)
    except Exception:
        mps_backend = None
    if mps_backend is not None:
        try:
            if mps_backend.is_available():
                return "local", "Apple MPS"
        except Exception:
            pass

    return "local", "CPU"


def _ordered_model_choices() -> List[Dict[str, object]]:
    download_root = getattr(tx, "WHISPER_CACHE", os.path.join(os.path.dirname(__file__), "models", "whisper"))
    choices: List[Dict[str, object]] = []
    for idx, (slug, label) in enumerate(MODEL_LIBRARY):
        model_path = os.path.join(download_root, f"{slug}.pt")
        downloaded = os.path.exists(model_path)
        choices.append(
            {
                "value": slug,
                "label": label,
                "downloaded": downloaded,
                "order": idx,
            }
        )

    choices.sort(key=lambda item: (not item["downloaded"], item["order"]))
    return choices


def _run_job(job: Job, *, mode: str, local_model: str, language: Optional[str],
             diarize: bool, aggregate: bool, timestamps: bool,
             mlx_batch_size: int = 12, mlx_quant: Optional[str] = None,
             trim_silence: bool = True):
    import logging
    logger = logging.getLogger("webapp._run_job")
    # Capture tx.log into the job while still printing to stdout
    orig_log = tx.log

    def set_progress(pct: float, label: Optional[str] = None):
        job.progress = max(job.progress, min(max(pct, 0.0), 1.0))
        if label:
            job.stage = label

    def web_log(msg: str):
        handled = False
        if msg.startswith("[PROGRESS]"):
            try:
                payload = json.loads(msg.split(" ", 1)[1])
            except Exception:
                payload = None
            if payload:
                tracker = job.tracker
                if tracker is None:
                    tracker = JobProgressTracker(
                        diarize=bool(job.options.get("diarize")),
                        mode=job.options.get("mode"),
                    )
                    job.tracker = tracker
                snapshot = tracker.update_from_payload(payload)
                if snapshot:
                    job.progress_meta = snapshot
                    job.stage_detail = snapshot.get("description", job.stage_detail)
                    set_progress(snapshot.get("overall", job.progress), snapshot.get("stage_label"))
                handled = True
        if not handled:
            try:
                job.log += (msg + "\n")
            except Exception:
                pass
            if msg.startswith("[SUCCESS]"):
                set_progress(1.0, "Transcript ready")
        # Also forward to console for debugging
        try:
            orig_log(msg)
        except Exception:
            pass

    tx.log = web_log  # monkey-patch

    try:
        logger.debug(f"[_run_job] Starting job {job.id} (infile={job.infile}, outfile={job.outfile})")
        # Serialize runs to keep logging sane
        run_lock.acquire()
        # Keep web default simple and robust: default to API unless user chose otherwise
        m = mode or "api"
        set_progress(0.1, "Starting transcription")

        if m == "local":
            resolved_mode, backend_label = _resolve_local_backend()
            m = resolved_mode
            web_log(f"[INFO] Local engine resolved → {backend_label}")
            if backend_label == "CPU":
                set_progress(0.12, "Running on CPU")
            elif backend_label == "NVIDIA GPU":
                set_progress(0.12, "Running on NVIDIA GPU")
            elif backend_label == "Apple MLX":
                set_progress(0.12, "Running on MLX")
            elif backend_label == "Apple MPS":
                set_progress(0.12, "Running on Apple MPS")
            else:
                set_progress(0.12, f"Running on {backend_label}")

        # Mirror CLI fallbacks so web behaves like terminal
        if m == "mlx" and not tx.is_apple_silicon():
            web_log("[WARN] MLX requested off Apple Silicon – falling back to API mode.")
            m = "api"
        if m == "mlx" and not getattr(tx, "LightningWhisperMLX", None):
            if getattr(tx, "whisper", None) is not None:
                web_log("[WARN] MLX requested but MLX backend unavailable – falling back to Local mode.")
                m = "local"
            else:
                web_log("[WARN] MLX and Local modes unavailable – falling back to API mode.")
                m = "api"
        elif m == "mlx" and getattr(tx, "MLX_BACKEND", None):
            web_log(f"[INFO] Using MLX backend: {tx.MLX_BACKEND}")

        tx.run(
            job.infile,
            job.outfile,
            mode=m,
            local_tag=local_model or "base",
            mlx_batch_size=mlx_batch_size,
            mlx_quant=mlx_quant,
            diarize=diarize,
            lang=(language or None),
            show_ts=timestamps,
            agg=aggregate,
            trim_silence=trim_silence,
        )
        logger.debug(f"[_run_job] tx.run finished for job {job.id}")
        if os.path.exists(job.outfile):
            logger.debug(f"[_run_job] Output file exists for job {job.id}: {job.outfile}")
        else:
            logger.error(f"[_run_job] Output file missing for job {job.id}: {job.outfile}")
        with open(job.outfile, "r", encoding="utf-8") as fh:
            job.result = fh.read()
        job.status = "done"
        logger.debug(f"[_run_job] Job {job.id} status set to 'done'")
        set_progress(1.0, "Transcript ready")
    except Exception as e:
        job.status = "error"
        logger.error(f"[_run_job] Job {job.id} failed: {e}")
        job.error = f"{e}\n\n" + traceback.format_exc()
        job.log += f"\n[ERROR] {e}\n"
        set_progress(1.0, "Processing failed")
    finally:
        # Restore logger
        tx.log = orig_log
        try:
            run_lock.release()
        except Exception:
            pass


@app.get("/")
def index():
    # Show MLX option on Apple Silicon, even if backend may fallback
    mlx_available = tx.is_apple_silicon() and bool(getattr(tx, "LightningWhisperMLX", None))
    model_choices = _ordered_model_choices()
    default_model = "base"
    default_model_label = next(
        (item["label"] for item in model_choices if item["value"] == default_model),
        model_choices[0]["label"] if model_choices else default_model,
    )

    return render_template(
        "index.html",
        mlx_available=mlx_available,
        model_choices=model_choices,
        default_model=default_model,
        default_model_label=default_model_label,
    )


@app.post("/start")
def start():
    f = request.files.get("audio")
    if not f:
        return ("No file uploaded", 400)

    # Options
    mode = request.form.get("mode") or None
    language = request.form.get("language") or None
    local_model = request.form.get("local_model") or "base"
    mlx_quant = request.form.get("mlx_quant") or None
    try:
        mlx_batch_size = int(request.form.get("mlx_batch_size") or 12)
    except Exception:
        mlx_batch_size = 12
    diarize = bool(request.form.get("diarize"))
    # Aggregation is always enabled in the web UI
    aggregate = True
    timestamps = bool(request.form.get("timestamps"))

    jid = uuid.uuid4().hex[:12]
    in_path = os.path.join(UPLOAD_DIR, f"{jid}_{f.filename}")
    out_path = os.path.join(OUT_DIR, f"{jid}.txt")
    f.save(in_path)

    job = Job(id=jid, infile=in_path, outfile=out_path)
    job.options.update(
        {
            "mode": mode,
            "language": language,
            "local_model": local_model,
            "mlx_batch_size": mlx_batch_size,
            "mlx_quant": mlx_quant,
            "diarize": diarize,
            "aggregate": aggregate,
            "timestamps": timestamps,
            "trim_silence": True,
        }
    )
    job.tracker = JobProgressTracker(diarize=diarize, mode=mode)
    job.progress_meta = job.tracker.snapshot()
    job.stage = job.progress_meta.get("stage_label", job.stage)
    job.stage_detail = job.progress_meta.get("description", job.stage_detail)
    job.progress = job.progress_meta.get("overall", job.progress)
    with jobs_lock:
        jobs[jid] = job

    t = threading.Thread(
        target=_run_job,
        args=(job,),
        kwargs=dict(
            mode=mode,
            local_model=local_model,
            language=language,
            diarize=diarize,
            aggregate=aggregate,
            timestamps=timestamps,
            mlx_quant=mlx_quant,
            mlx_batch_size=mlx_batch_size,
            trim_silence=True,
        ),
        daemon=True,
    )
    t.start()

    # Serve JSON to clients using fetch/XHR, fall back to legacy redirect otherwise
    wants_json = request.accept_mimetypes.best == "application/json" or (
        request.accept_mimetypes["application/json"]
        and request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]
    ) or request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if wants_json:
        return jsonify(
            {
                "job_id": jid,
                "logs_url": url_for("logs", job_id=jid),
                "download_url": url_for("download", job_id=jid),
                "progress": job.progress,
                "stage": job.stage,
            }
        )

    return redirect(url_for("status", job_id=jid))


@app.get("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return ("Not found", 404)
    return render_template(
        "status.html",
        job_id=job_id,
        initial_meta=job.progress_meta,
        initial_stage=job.stage,
        initial_description=job.stage_detail,
        initial_progress=job.progress,
        job_status=job.status,
    )


@app.get("/logs/<job_id>")
def logs(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    payload = {
        "status": job.status,
        "log": job.log,
        "progress": job.progress,
        "stage": job.stage,
        "description": job.stage_detail,
        "stageKey": job.progress_meta.get("stage_key"),
        "stageProgress": job.progress_meta.get("stage_progress"),
        "timeline": job.progress_meta.get("timeline", []),
    }
    if job.status == "done":
        payload["result"] = job.result or ""
    if job.status == "error":
        payload["error"] = job.error
    return jsonify(payload)


@app.get("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or not os.path.exists(job.outfile):
        return ("Not found", 404)
    return send_file(
        job.outfile,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"transcript_{job_id}.txt",
    )


if __name__ == "__main__":
    # Bind to all interfaces for convenience in local networks
    app.run(host="0.0.0.0", port=5001, debug=True)
