#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

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


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()
run_lock = threading.Lock()


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "web_runs", "uploads")
OUT_DIR = os.path.join(os.path.dirname(__file__), "web_runs", "outputs")
_ensure_dir(UPLOAD_DIR)
_ensure_dir(OUT_DIR)


def _run_job(job: Job, *, mode: str, local_model: str, language: Optional[str],
             diarize: bool, aggregate: bool, timestamps: bool,
             mlx_batch_size: int = 12, mlx_quant: Optional[str] = None):
    # Capture tx.log into the job while still printing to stdout
    orig_log = tx.log

    def web_log(msg: str):
        try:
            job.log += (msg + "\n")
        except Exception:
            pass
        # Also forward to console for debugging
        try:
            orig_log(msg)
        except Exception:
            pass

    tx.log = web_log  # monkey-patch

    try:
        # Serialize runs to keep logging sane
        run_lock.acquire()
        # Keep web default simple and robust: default to API unless user chose otherwise
        m = mode or "api"

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
        )

        with open(job.outfile, "r", encoding="utf-8") as fh:
            job.result = fh.read()
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = f"{e}\n\n" + traceback.format_exc()
        job.log += f"\n[ERROR] {e}\n"
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
    return render_template("index.html", mlx_available=mlx_available)


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
    aggregate = bool(request.form.get("aggregate"))
    timestamps = bool(request.form.get("timestamps"))

    jid = uuid.uuid4().hex[:12]
    in_path = os.path.join(UPLOAD_DIR, f"{jid}_{f.filename}")
    out_path = os.path.join(OUT_DIR, f"{jid}.txt")
    f.save(in_path)

    job = Job(id=jid, infile=in_path, outfile=out_path)
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
        ),
        daemon=True,
    )
    t.start()

    return redirect(url_for("status", job_id=jid))


@app.get("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return ("Not found", 404)
    return render_template("status.html", job_id=job_id)


@app.get("/logs/<job_id>")
def logs(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    payload = {
        "status": job.status,
        "log": job.log,
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
