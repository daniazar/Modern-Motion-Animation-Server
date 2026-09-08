import os
import sys
import time

sys.path.insert(0, "/app/server")
sys.path.insert(0, "/app/engines")

from engines.animate_anyone_engine import animate_anyone_engine

ref_image = "/app/public/avatars/master_ruby_archive/chroma_master_poses/ruby_02a_scarlet_corset_default_clasped.jpg"
drv_video = "/app/public/assets/action_loops/ruby_master_suite/ruby_gesture_left.mp4"
out_video = "/app/storage/motion/outputs/ruby_corset_gesture_left_full.mp4"

print(f"Checking files:")
print(f"  Ref image exists: {os.path.exists(ref_image)} ({ref_image})")
print(f"  Drv video exists: {os.path.exists(drv_video)} ({drv_video})")
print(f"  Engine available: {animate_anyone_engine.is_available()}")

if not animate_anyone_engine.is_available():
    print("ERROR: Weights missing!")
    sys.exit(1)

print("Starting animation of Ruby in Scarlet Corset with ruby_gesture_left.mp4...")
t0 = time.time()
res = animate_anyone_engine.generate(
    image_path=ref_image,
    driving_video_path=drv_video,
    output_path=out_video,
    num_frames=64,
    steps=20,
    cfg=3.5,
    seed=42,
    width=512,
    height=768,
)
print(f"Finished in {round(time.time() - t0, 1)}s!")
print(res)
