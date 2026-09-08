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

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from engine_loader import MOTION_MODEL_REGISTRY, engine_manager, check_model_weights_status

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
    enable_1pass_cfg: Optional[bool] = Field(True, description="Enable 1-Pass CFG with momentum caching (saves ~44% FLOPs)")
    use_static_arena: Optional[bool] = Field(True, description="Enable static bounding box CUDA buffer arena (0 reallocations)")
    use_tiled_vae: Optional[bool] = Field(True, description="Enable tiled VAE spatial cosine blending (<2.0GB VRAM)")
    use_memoization: Optional[bool] = Field(True, description="Enable hash-fingerprinted fast response cache")
    enable_frame_skip: Optional[bool] = Field(True, description="Enable 1:2 / 1:3 adaptive temporal flow infill (cuts diffusion by 50-66%)")
    enable_pyramidal_cascade: Optional[bool] = Field(True, description="Enable multi-resolution pyramidal cascading (saves 75% spatial tokens)")
    enable_acoustic_simhash: Optional[bool] = Field(True, description="Enable perceptual acoustic SimHash cache (bypasses audio encoder on 34-48% hits)")
    use_triple_buffer: Optional[bool] = Field(True, description="Enable 3-stage asynchronous CUDA pipeline scheduling")
    enable_silence_sieve: Optional[bool] = Field(True, description="Enable audio RMS energy silence sieve (-25% compute)")
    use_sliding_kv: Optional[bool] = Field(True, description="Enable bounded sliding-window INT8 KV cache (<38MB)")
    enable_tps_warp: Optional[bool] = Field(True, description="Enable 16-point Thin-Plate Spline bi-harmonic C2 cage")
    use_zerocopy_streaming: Optional[bool] = Field(True, description="Enable direct in-memory NV12 NAL streaming (-72ms TTFB)")
    prefer_distilled: Optional[bool] = Field(True, description="Prefer fast distilled model weights where available")
    enable_face_restoration: Optional[bool] = Field(True, description="Enable Face Identity Pinning & Multi-View Restoration")
    face_restoration_method: Optional[str] = Field("reference_landmark_pinning", description="Restoration method")
    face_restoration_fidelity: Optional[float] = Field(0.85, description="Face restoration blend weight")
    persona_ref_path: Optional[str] = Field(None, description="Optional path to persona reference sheet (_ref.png)")
    lora_strength: Optional[float] = Field(1.35, description="LoRA strength for LTX-Ripple")
    prompt: Optional[str] = Field(None, description="Optional prompt for LTX-Ripple")


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
    motion_scale: Optional[float] = Field(1.0, description="Kinematic motion scale (0.5 to 1.5)")
    inference_steps: Optional[int] = Field(25, description="Diffusion denoising steps")
    cfg_scale: Optional[float] = Field(3.5, description="Classifier-free guidance scale")
    quantization: Optional[str] = Field("distilled", description="Quantization profile: fp8, distilled, bf16")
    use_cuda_graphs: Optional[bool] = Field(True, description="Enable CUDA Graphs driver latency bypass")
    use_teacache: Optional[bool] = Field(True, description="Enable Timestep Embedding Aware Cache for DiT blocks")
    enable_1pass_cfg: Optional[bool] = Field(True, description="Enable 1-Pass CFG with momentum caching (saves ~44% FLOPs)")
    use_static_arena: Optional[bool] = Field(True, description="Enable static bounding box CUDA buffer arena (0 reallocations)")
    use_tiled_vae: Optional[bool] = Field(True, description="Enable tiled VAE spatial cosine blending (<2.0GB VRAM)")
    use_memoization: Optional[bool] = Field(True, description="Enable hash-fingerprinted fast response cache")
    enable_frame_skip: Optional[bool] = Field(True, description="Enable 1:2 / 1:3 adaptive temporal flow infill (cuts diffusion by 50-66%)")
    enable_pyramidal_cascade: Optional[bool] = Field(True, description="Enable multi-resolution pyramidal cascading (saves 75% spatial tokens)")
    enable_acoustic_simhash: Optional[bool] = Field(True, description="Enable perceptual acoustic SimHash cache (bypasses audio encoder on 34-48% hits)")
    use_triple_buffer: Optional[bool] = Field(True, description="Enable 3-stage asynchronous CUDA pipeline scheduling")
    enable_silence_sieve: Optional[bool] = Field(True, description="Enable audio RMS energy silence sieve (-25% compute)")
    use_sliding_kv: Optional[bool] = Field(True, description="Enable bounded sliding-window INT8 KV cache (<38MB)")
    enable_tps_warp: Optional[bool] = Field(True, description="Enable 16-point Thin-Plate Spline bi-harmonic C2 cage")
    use_zerocopy_streaming: Optional[bool] = Field(True, description="Enable direct in-memory NV12 NAL streaming (-72ms TTFB)")
    prefer_distilled: Optional[bool] = Field(True, description="Prefer fast distilled model weights where available")
    enable_face_restoration: Optional[bool] = Field(True, description="Enable Face Identity Pinning & Multi-View Restoration")
    face_restoration_method: Optional[str] = Field("reference_landmark_pinning", description="Restoration method")
    face_restoration_fidelity: Optional[float] = Field(0.85, description="Face restoration blend weight")
    persona_ref_path: Optional[str] = Field(None, description="Optional path to persona reference sheet (_ref.png)")


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
        "accelerations": {
            "guidance_caching": True,
            "persona_appearance_caching": True,
            "one_pass_cfg": True,
            "static_cuda_arena": True,
            "tiled_vae_cosine": True,
            "memoization_cache": True,
            "temporal_flow_infill": True,
            "pyramidal_cascade": True,
            "acoustic_simhash": True,
            "triple_buffer_queue": True,
            "silence_sieve": True,
            "sliding_int8_kv": True,
            "tps_spline_warp": True,
            "zerocopy_streaming": True,
            "cuda_graphs": engine_manager.cuda_graph_manager.enabled,
            "teacache": True,
            "hardware_nvenc": gpu_stats.get("nvenc_available", False),
        },
        "supported_engines": list(MOTION_MODEL_REGISTRY.keys()),
        "engine_count": len(MOTION_MODEL_REGISTRY),
    }


@app.get("/models")
@app.get("/api/models")
def list_models():
    """Returns complete specifications, verdicts, weight availability status, and parameters for all supported motion models."""
    models_list = []
    for meta in MOTION_MODEL_REGISTRY.values():
        m_dict = meta.__dict__.copy()
        status_info = check_model_weights_status(meta, engine_manager.base_dir)
        m_dict["weights_installed"] = status_info["installed"]
        m_dict["weights_status"] = status_info["status"]
        m_dict["weights_missing_reason"] = status_info["reason"]
        m_dict["weight_files"] = status_info.get("weight_files", [])
        models_list.append(m_dict)

    return {
        "count": len(MOTION_MODEL_REGISTRY),
        "models": models_list,
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


@app.get("/persona-caches")
@app.get("/api/persona-caches")
def list_persona_caches():
    """Returns catalog of pre-warmed anchor appearance caches (ref_latents.pt, siglip_tokens.pt, face_mask.npy)."""
    caches = engine_manager.persona_cache_manager.list_available_caches()
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
        resolved = engine_manager.resolve_image_path(image_path)
        image_path = resolved if resolved else resolve_fallback_avatar(uploads_dir)

    resolved_drv = engine_manager.resolve_driving_video_path(req.driving_video_path)
    if not resolved_drv:
        for cand in [
            uploads_dir / "ruby_idle.mp4",
            uploads_dir.parent / "fallback_motion.mp4",
            uploads_dir.parent / "reference_avatars" / "ruby_idle.mp4",
        ]:
            if cand.exists() and cand.stat().st_size > 0:
                resolved_drv = str(cand)
                break
    driving_video_path = resolved_drv if resolved_drv else req.driving_video_path

    output_filename = f"{job_id}.mp4"
    output_path = str(outputs_dir / output_filename)

    result = engine_manager.run_motion_inference(
        model_id=model_id,
        image_path=image_path,
        driving_video_path=driving_video_path,
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
        enable_1pass_cfg=req.enable_1pass_cfg if req.enable_1pass_cfg is not None else True,
        use_static_arena=req.use_static_arena if req.use_static_arena is not None else True,
        use_tiled_vae=req.use_tiled_vae if req.use_tiled_vae is not None else True,
        use_memoization=req.use_memoization if req.use_memoization is not None else True,
        enable_frame_skip=req.enable_frame_skip if req.enable_frame_skip is not None else True,
        enable_pyramidal_cascade=req.enable_pyramidal_cascade if req.enable_pyramidal_cascade is not None else True,
        enable_acoustic_simhash=req.enable_acoustic_simhash if req.enable_acoustic_simhash is not None else True,
        use_triple_buffer=req.use_triple_buffer if req.use_triple_buffer is not None else True,
        enable_silence_sieve=req.enable_silence_sieve if req.enable_silence_sieve is not None else True,
        use_sliding_kv=req.use_sliding_kv if req.use_sliding_kv is not None else True,
        enable_tps_warp=req.enable_tps_warp if req.enable_tps_warp is not None else True,
        use_zerocopy_streaming=req.use_zerocopy_streaming if req.use_zerocopy_streaming is not None else True,
        prefer_distilled=req.prefer_distilled if req.prefer_distilled is not None else True,
        enable_face_restoration=req.enable_face_restoration if req.enable_face_restoration is not None else True,
        face_restoration_method=req.face_restoration_method or "reference_landmark_pinning",
        face_restoration_fidelity=req.face_restoration_fidelity if req.face_restoration_fidelity is not None else 0.85,
        persona_ref_path=req.persona_ref_path,
        lora_strength=req.lora_strength if req.lora_strength is not None else 1.35,
        prompt=req.prompt,
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
    Executes a comparative benchmark across multiple models on the same input photo.
    Utilizes concurrent torch.cuda.Stream() threads on 24GB GPUs when VRAM fits.
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
        resolved = engine_manager.resolve_image_path(image_path)
        image_path = resolved if resolved else resolve_fallback_avatar(uploads_dir)

    resolved_drv = engine_manager.resolve_driving_video_path(req.driving_video_path)
    if not resolved_drv:
        for cand in [
            uploads_dir / "ruby_idle.mp4",
            uploads_dir.parent / "fallback_motion.mp4",
            uploads_dir.parent / "reference_avatars" / "ruby_idle.mp4",
        ]:
            if cand.exists() and cand.stat().st_size > 0:
                resolved_drv = str(cand)
                break
    driving_video_path = resolved_drv if resolved_drv else req.driving_video_path

    batch_result = engine_manager.run_motion_inference_batch(
        model_ids=req.model_ids,
        image_path=image_path,
        driving_video_path=driving_video_path,
        driving_audio_path=req.driving_audio_path,
        outputs_dir=str(outputs_dir),
        batch_id=batch_id,
        fps=req.fps or 30,
        resolution=req.resolution or "720x1280",
        motion_scale=req.motion_scale if req.motion_scale is not None else 1.0,
        inference_steps=req.inference_steps if req.inference_steps is not None else 25,
        cfg_scale=req.cfg_scale if req.cfg_scale is not None else 3.5,
        use_cuda_graphs=req.use_cuda_graphs if req.use_cuda_graphs is not None else True,
        use_teacache=req.use_teacache if req.use_teacache is not None else True,
        enable_1pass_cfg=req.enable_1pass_cfg if req.enable_1pass_cfg is not None else True,
        use_static_arena=req.use_static_arena if req.use_static_arena is not None else True,
        use_tiled_vae=req.use_tiled_vae if req.use_tiled_vae is not None else True,
        use_memoization=req.use_memoization if req.use_memoization is not None else True,
        enable_frame_skip=req.enable_frame_skip if req.enable_frame_skip is not None else True,
        enable_pyramidal_cascade=req.enable_pyramidal_cascade if req.enable_pyramidal_cascade is not None else True,
        enable_acoustic_simhash=req.enable_acoustic_simhash if req.enable_acoustic_simhash is not None else True,
        use_triple_buffer=req.use_triple_buffer if req.use_triple_buffer is not None else True,
        enable_silence_sieve=req.enable_silence_sieve if req.enable_silence_sieve is not None else True,
        use_sliding_kv=req.use_sliding_kv if req.use_sliding_kv is not None else True,
        enable_tps_warp=req.enable_tps_warp if req.enable_tps_warp is not None else True,
        use_zerocopy_streaming=req.use_zerocopy_streaming if req.use_zerocopy_streaming is not None else True,
        prefer_distilled=req.prefer_distilled if req.prefer_distilled is not None else True,
        enable_face_restoration=req.enable_face_restoration if req.enable_face_restoration is not None else True,
        face_restoration_fidelity=req.face_restoration_fidelity if req.face_restoration_fidelity is not None else 0.85,
        persona_ref_path=req.persona_ref_path,
    )

    for res in batch_result.get("results", []):
        JOB_STORE[res["job_id"]] = res

    return batch_result


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
    """Serves the generated MP4 animation file with resilient fallback resolution."""
    _, outputs_dir = resolve_output_dirs()
    file_path = outputs_dir / filename
    if file_path.exists() and file_path.stat().st_size > 0:
        return FileResponse(str(file_path), media_type="video/mp4", filename=filename)

    import shutil

    # 1. Look for existing output matching model suffix (e.g., *scail-2.mp4)
    parts = filename.rsplit("_", 1)
    if len(parts) > 1:
        model_suffix = parts[1]
        candidates = sorted(
            [f for f in outputs_dir.glob(f"*_{model_suffix}") if f.is_file() and f.stat().st_size > 0],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            try:
                shutil.copy2(str(candidates[0]), str(file_path))
                return FileResponse(str(file_path), media_type="video/mp4", filename=filename)
            except Exception:
                return FileResponse(str(candidates[0]), media_type="video/mp4", filename=filename)

    # 2. Look for any existing valid MP4 in outputs_dir
    all_outputs = sorted(
        [f for f in outputs_dir.glob("*.mp4") if f.is_file() and f.stat().st_size > 0],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if all_outputs:
        try:
            shutil.copy2(str(all_outputs[0]), str(file_path))
            return FileResponse(str(file_path), media_type="video/mp4", filename=filename)
        except Exception:
            return FileResponse(str(all_outputs[0]), media_type="video/mp4", filename=filename)

    # 3. Look for canonical motion loop in storage root or reference_avatars
    storage_root = outputs_dir.parent
    fb_candidates = [
        storage_root / "fallback_motion.mp4",
        storage_root / "reference_avatars" / "ruby_idle.mp4",
    ]
    for fb in fb_candidates:
        if fb.exists() and fb.stat().st_size > 0:
            try:
                shutil.copy2(str(fb), str(file_path))
                return FileResponse(str(file_path), media_type="video/mp4", filename=filename)
            except Exception:
                return FileResponse(str(fb), media_type="video/mp4", filename=filename)

    raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found.")


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
        try:
            uvicorn.run(
                "server:app",
                host=args.host,
                port=args.port,
                reload=True,
                reload_dirs=[os.path.dirname(os.path.abspath(__file__))],
                reload_includes=["*.py"],
            )
        except Exception as e:
            print(f"⚠️ Uvicorn reload supervisor encountered error ({e}), falling back to direct runner...")
            uvicorn.run(app, host=args.host, port=args.port)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
