#!/usr/bin/env python3
"""
FastAPI endpoint for the Joint Gravity-Magnetic Inversion pipeline.

All input files are passed as server-side paths (mounted into the container).
Long-running jobs are executed in a background thread and tracked by job ID.

Endpoints
---------
POST /inversion/run          — Submit a new inversion job
GET  /inversion/status/{id}  — Poll job status and retrieve results path
GET  /health                 — Liveness probe
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import traceback
import uuid
from enum import Enum
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Joint Gravity-Magnetic Inversion API",
    description="Physics-informed U-Net inversion for gravity and magnetic data.",
    version="1.0.0",
)

log = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
#  Job store  (in-memory; replace with Redis/DB for multi-worker deployments)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobRecord(BaseModel):
    job_id:     str
    status:     JobStatus
    output_dir: Optional[str] = None
    error:      Optional[str] = None


_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
#  Request / Response schemas
# ---------------------------------------------------------------------------

class InversionRequest(BaseModel):
    # ── Required data inputs ─────────────────────────────────────────────
    grav:    str = Field(..., description="Absolute path to Bouguer anomaly NetCDF (.nc)")
    rtp_mag: str = Field(..., description="Absolute path to RTP magnetic NetCDF (.nc)")
    mag:     str = Field(..., description="Absolute path to raw magnetic NetCDF (.nc)")
    dem:     str = Field(..., description="Absolute path to DEM NetCDF (.nc)")

    # ── Model geometry ───────────────────────────────────────────────────
    coarsen_factor: int   = Field(1,    ge=1, description="Coarsen factor applied to all grids")
    depth:          float = Field(3000, gt=0, description="Total model depth (metres)")

    # ── Training ─────────────────────────────────────────────────────────
    num_iterations: int   = Field(5000, ge=1,   description="Number of training iterations")
    lr:             float = Field(1e-3, gt=0.0, description="Peak learning rate")

    # ── Optional well-log constraints ────────────────────────────────────
    las_dir:   Optional[str] = Field(None, description="Directory containing .las files")
    json_path: Optional[str] = Field(None, description="JSON file with well locations")

    # ── Optional seismic constraints ─────────────────────────────────────
    seismic_2d: Optional[str] = Field(None, description="Path to 2-D FWI HDF5 file")
    seismic_3d: Optional[str] = Field(None, description="Path to 3-D reflection HDF5 file")

    # ── Output ───────────────────────────────────────────────────────────
    output_dir:  str = Field("Output/Inversion/Joint",    description="Directory to write outputs")
    output_name: str = Field("joint_inversion_results",   description="Base filename (no extension)")


class InversionResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    message: str


# ---------------------------------------------------------------------------
#  Background worker
# ---------------------------------------------------------------------------

def _run_inversion(job_id: str, req: InversionRequest) -> None:
    """Execute the inversion pipeline in a background thread."""
    from inversion_model import main as inversion_main  # lazy import

    with _jobs_lock:
        _jobs[job_id].status = JobStatus.RUNNING

    try:
        args = argparse.Namespace(
            grav           = req.grav,
            rtp_mag        = req.rtp_mag,
            mag            = req.mag,
            dem            = req.dem,
            coarsen_factor = req.coarsen_factor,
            depth          = req.depth,
            num_iterations = req.num_iterations,
            lr             = req.lr,
            las_dir        = req.las_dir,
            json_path      = req.json_path,
            seismic_2d     = req.seismic_2d,
            seismic_3d     = req.seismic_3d,
            output_dir     = req.output_dir,
            output_name    = req.output_name,
        )
        inversion_main(args)

        with _jobs_lock:
            _jobs[job_id].status     = JobStatus.COMPLETED
            _jobs[job_id].output_dir = req.output_dir

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].status = JobStatus.FAILED
            _jobs[job_id].error  = traceback.format_exc()
        log.error("Job %s failed: %s", job_id, exc)


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/inversion/run", response_model=InversionResponse, status_code=202, tags=["Inversion"])
def run_inversion(req: InversionRequest):
    """
    Submit a joint gravity-magnetic inversion job.

    Input files must be accessible on the server filesystem (use Docker volume
    mounts to expose your data directory). The job runs asynchronously;
    poll `/inversion/status/{job_id}` to track progress.
    """
    # Basic path validation — prevents path-traversal and missing-file errors early
    for field, path in [
        ("grav",    req.grav),
        ("rtp_mag", req.rtp_mag),
        ("mag",     req.mag),
        ("dem",     req.dem),
    ]:
        if not os.path.isfile(path):
            raise HTTPException(status_code=422, detail=f"File not found for '{field}': {path}")

    if req.las_dir and not os.path.isdir(req.las_dir):
        raise HTTPException(status_code=422, detail=f"LAS directory not found: {req.las_dir}")
    if req.json_path and not os.path.isfile(req.json_path):
        raise HTTPException(status_code=422, detail=f"JSON file not found: {req.json_path}")
    if req.seismic_2d and not os.path.isfile(req.seismic_2d):
        raise HTTPException(status_code=422, detail=f"Seismic 2D file not found: {req.seismic_2d}")
    if req.seismic_3d and not os.path.isfile(req.seismic_3d):
        raise HTTPException(status_code=422, detail=f"Seismic 3D file not found: {req.seismic_3d}")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = JobRecord(job_id=job_id, status=JobStatus.PENDING)

    thread = threading.Thread(target=_run_inversion, args=(job_id, req), daemon=True)
    thread.start()

    log.info("Job %s submitted", job_id)
    return InversionResponse(
        job_id  = job_id,
        status  = JobStatus.PENDING,
        message = f"Job submitted. Poll /inversion/status/{job_id} for updates.",
    )


@app.get("/inversion/status/{job_id}", response_model=JobRecord, tags=["Inversion"])
def get_job_status(job_id: str):
    """
    Retrieve the current status of an inversion job.

    Returns `output_dir` when the job completes, or `error` when it fails.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.get("/inversion/download/{job_id}", tags=["Inversion"])
def download_result(job_id: str, filename: str):
    """
    Download a specific output file from a completed job.

    Pass the base filename (e.g. `joint_inversion_results.npz`) as the
    `filename` query parameter.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet (status={job.status})")

    # Sanitise filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(job.output_dir, safe_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Output file not found: {safe_filename}")

    return FileResponse(path=file_path, filename=safe_filename)


# ---------------------------------------------------------------------------
#  Entry point  (uvicorn used in Docker CMD)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("endpoint:app", host="0.0.0.0", port=8000, reload=False)
