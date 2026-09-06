"""
vendor/Modern-Motion-Animation-Server/server.py

High-Throughput Modern Motion Animation Server Microservice (Port 8011).
Coordinates and benchmarks:
  - Wan-Animate-2
  - SCAIL-2
  - EchoMimicV3
  - EMOSH
  - MimicMotion
  - Animate Anyone (MooreThreads)
"""

import os
import sys
import time
import uuid
import base64
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine_loader import MOTION_MODEL_REGISTRY, engine_manager

PORT = int(os.environ.get("MODERN_MOTION_PORT", 8011))

app = FastAPI(
    title="NewsStudio Modern Motion Animation Server",
    description="Unified API & Container for Next-Gen Human Body, Arms, and Hands Motion Retargeting.",
    version="1.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
JOB_STORE: Dict[str, Dict[str, Any]] = {}


class AnimateRequest(BaseModel):
    model_id: str = Field(..., description="Target model ID: wan-animate-2, scail-2, echomimicv3, emosh, mimicmotion, animate-anyone")
    image_path: Optional[str] = Field(None, description="Absolute or relative path to chroma reference photo")
    image_base64: Optional[str] = Field(None, description="Optional base64-encoded reference photo")
    driving_video_path: Optional[str] = Field(None, description="Path to driving motion clip (.mp4)")
    driving_audio_path: Optional[str] = Field(None, description="Path to driving speech audio (.wav for EchoMimicV3)")
    fps: Optional[int] = Field(30, description="Target frame rate")
    resolution: Optional[str] = Field("720x1280", description="Output resolution (e.g., 720x1280, 512x512)")
    motion_scale: Optional[float] = Field(1.0, description="Motion amplitude scale (0.5 to 1.5)")
    inference_steps: Optional[int] = Field(25, description="Diffusion denoising steps")
    cfg_scale: Optional[float] = Field(3.5, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(42, description="Random seed")
    use_cuda_graphs: Optional[bool] = Field(True, description="Enable CUDA Graphs driver latency bypass")
    use_teacache: Optional[bool] = Field(True, description="Enable Timestep Embedding Aware Cache for DiT blocks")


class CompareRequest(BaseModel):
    model_ids: List[str] = Field(
        default=["wan-animate-2", "scail-2", "echomimicv3", "mimicmotion"],
        description="List of models to benchmark in parallel/sequence"
    )
    image_path: Optional[str] = Field(None, description="Chroma reference photo path")
    image_base64: Optional[str] = Field(None, description="Optional base64 photo")
    driving_video_path: Optional[str] = Field(None, description="Driving motion clip")
    driving_audio_path: Optional[str] = Field(None, description="Driving speech audio")
    fps: Optional[int] = Field(30, description="Target frame rate")
    resolution: Optional[str] = Field("720x1280", description="Target resolution")
    use_cuda_graphs: Optional[bool] = Field(True, description="Enable CUDA Graphs driver latency bypass")
    use_teacache: Optional[bool] = Field(True, description="Enable Timestep Embedding Aware Cache for DiT blocks")


def resolve_output_dirs() -> Tuple[Path, Path]:
    candidates = [
        Path("/app/storage/motion"),
        Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / "storage" / "motion",
        Path(os.getcwd()) / "storage" / "motion",
    ]
    storage_root = candidates[0]
    for c in candidates:
        if c.exists():
            storage_root = c
            break
    uploads_dir = storage_root / "uploads"
    outputs_dir = storage_root / "outputs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir, outputs_dir


def resolve_fallback_avatar(uploads_dir: Path) -> str:
    storage_root = uploads_dir.parent
    candidates = [
        storage_root / "reference_avatars" / "ruby_body_ref.png",
        storage_root / "reference_avatars" / "ruby.png",
        Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / "public" / "avatars" / "ruby_body_ref.png",
        Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / "public" / "avatars" / "ruby.png",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    fallback = uploads_dir / "fallback_chroma.png"
    if not fallback.exists():
        try:
            from PIL import Image
            img = Image.new("RGB", (720, 1280), color=(0, 255, 0))
            img.save(str(fallback))
            return str(fallback)
        except Exception:
            pass
    return str(fallback)


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Returns microservice health, GPU memory usage, and registered engines."""
    gpu_stats = engine_manager.get_gpu_telemetry()
    return {
        "status": "online",
        "service": "modern-motion-animation-server",
        "port": PORT,
        "active_model": engine_manager.active_model_id,
        "gpu": gpu_stats,
        "supported_engines": list(MOTION_MODEL_REGISTRY.keys()),
        "engine_count": len(MOTION_MODEL_REGISTRY),
    }


@app.get("/models")
@app.get("/api/models")
def list_models():
    """Returns complete specifications, verdicts, and parameters for all supported motion models."""
    return {
        "count": len(MOTION_MODEL_REGISTRY),
        "models": [meta.__dict__ for meta in MOTION_MODEL_REGISTRY.values()],
    }


@app.get("/caches")
@app.get("/api/caches")
def list_guidance_caches():
    """Returns catalog of pre-vectorized guidance caches (DWPose, SMPL-X, VAE Latents) ready for 0ms loading."""
    caches = engine_manager.cache_manager.list_available_caches()
    return {
        "count": len(caches),
        "caches": caches,
    }


@app.post("/animate")
@app.post("/api/animate")
async def animate_photo(req: AnimateRequest):
    """
    Animates a green chroma reference photo with the specified model and driving motion.
    """
    model_id = req.model_id.lower()
    if model_id not in MOTION_MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id: '{req.model_id}'. Supported: {list(MOTION_MODEL_REGISTRY.keys())}",
        )

    uploads_dir, outputs_dir = resolve_output_dirs()
    timestamp = int(time.time() * 1000)
    job_id = f"motion_{model_id}_{timestamp}"

    # Handle image path
    image_path = req.image_path
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        image_path = str(uploads_dir / f"ref_{timestamp}.png")
        with open(image_path, "wb") as f:
            f.write(img_bytes)

    if not image_path or not os.path.exists(image_path):
        image_path = resolve_fallback_avatar(uploads_dir)

    output_filename = f"{job_id}.mp4"
    output_path = str(outputs_dir / output_filename)

    result = engine_manager.run_motion_inference(
        model_id=model_id,
        image_path=image_path,
        driving_video_path=req.driving_video_path,
        driving_audio_path=req.driving_audio_path,
        output_path=output_path,
        fps=req.fps or 30,
        resolution=req.resolution or "720x1280",
        motion_scale=req.motion_scale or 1.0,
        inference_steps=req.inference_steps or 25,
        cfg_scale=req.cfg_scale or 3.5,
        seed=req.seed or 42,
        use_cuda_graphs=req.use_cuda_graphs if req.use_cuda_graphs is not None else True,
        use_teacache=req.use_teacache if req.use_teacache is not None else True,
    )

    result["job_id"] = job_id
    result["output_url"] = f"/api/motion/outputs/{output_filename}"
    result["filename"] = output_filename
    JOB_STORE[job_id] = result

    return result


@app.post("/compare")
@app.post("/api/compare")
async def compare_models(req: CompareRequest):
    """
    Executes a side-by-side comparative benchmark across multiple models on the same input photo.
    """
    if not req.model_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one model_id in model_ids list.")

    uploads_dir, outputs_dir = resolve_output_dirs()
    timestamp = int(time.time() * 1000)
    batch_id = f"compare_{timestamp}"

    # Handle reference image
    image_path = req.image_path
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        image_path = str(uploads_dir / f"ref_compare_{timestamp}.png")
        with open(image_path, "wb") as f:
            f.write(img_bytes)

    if not image_path or not os.path.exists(image_path):
        image_path = resolve_fallback_avatar(uploads_dir)

    results = []
    for model_id in req.model_ids:
        mid = model_id.lower()
        if mid not in MOTION_MODEL_REGISTRY:
            continue

        job_id = f"{batch_id}_{mid}"
        output_filename = f"{job_id}.mp4"
        output_path = str(outputs_dir / output_filename)

        res = engine_manager.run_motion_inference(
            model_id=mid,
            image_path=image_path,
            driving_video_path=req.driving_video_path,
            driving_audio_path=req.driving_audio_path,
            output_path=output_path,
            fps=req.fps or 30,
            resolution=req.resolution or "720x1280",
            use_cuda_graphs=req.use_cuda_graphs if req.use_cuda_graphs is not None else True,
            use_teacache=req.use_teacache if req.use_teacache is not None else True,
        )
        res["job_id"] = job_id
        res["output_url"] = f"/api/motion/outputs/{output_filename}"
        res["filename"] = output_filename
        JOB_STORE[job_id] = res
        results.append(res)

    return {
        "batch_id": batch_id,
        "input_image": image_path,
        "driving_video": req.driving_video_path,
        "models_evaluated": len(results),
        "results": results,
    }


@app.get("/jobs/{job_id}")
@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Retrieves current status, metrics, and video URL for a motion animation job."""
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOB_STORE[job_id]


@app.get("/outputs/{filename}")
@app.get("/api/outputs/{filename}")
def get_output_file(filename: str):
    """Serves the generated MP4 animation file."""
    _, outputs_dir = resolve_output_dirs()
    file_path = outputs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found.")
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("MODERN_MOTION_RELOAD", "false").lower() in ("true", "1", "yes"),
        help="Auto-reload on code change without container restart",
    )
    args = parser.parse_args()

    print(f"🚀 Starting Modern Motion Animation Server on {args.host}:{args.port} (reload={args.reload})")
    if args.reload:
        uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
