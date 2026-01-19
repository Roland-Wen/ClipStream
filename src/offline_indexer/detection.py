#!/usr/bin/env python3
"""
ClipStream - Scene Detection & Extraction Pipeline (CLI)
Author: Roland Wen
Description: 
    Production-grade script to process videos into semantic scenes.
    Supports resumable processing, batch lists, and environment detection.

Usage:
    python detection.py --video "episode_01.mp4"
    python detection.py --video_list "queue.txt" --threshold 5.0
"""

import os
import time
import json
import argparse
import sys
from typing import List, Dict, Any, Optional

# Conditional imports for environment detection
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# Try importing Computer Vision libs; warn if missing
try:
    import cv2
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector
except ImportError:
    print("❌ Critical Error: Missing dependencies.")
    print("   Please run: pip install scenedetect[opencv] opencv-python")
    sys.exit(1)


# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

class ProjectConfig:
    def __init__(self, base_path: str):
        self.base = base_path
        self.videos = os.path.join(base_path, "videos")
        self.scenes = os.path.join(base_path, "scenes")
        self.metadata = os.path.join(base_path, "metadata")
        
        # Ensure directories exist
        for p in [self.videos, self.scenes, self.metadata]:
            os.makedirs(p, exist_ok=True)

    def get_paths(self, video_filename: str) -> Dict[str, str]:
        """Generates consistent file paths for a specific video."""
        # Clean filename to remove paths if user provided full path
        filename = os.path.basename(video_filename)
        stem = os.path.splitext(filename)[0]
        
        return {
            "video_path": os.path.join(self.videos, filename),
            "img_dir": os.path.join(self.scenes, stem),
            "meta_file": os.path.join(self.metadata, f"{stem}_metadata.json"),
            "cuts_file": os.path.join(self.metadata, f"{stem}_cuts_cache.json"),
            "video_id": stem,
            "filename": filename
        }

def setup_environment(user_base_path: str) -> ProjectConfig:
    """Detects environment and sets up storage paths."""
    
    # Auto-mount if in Colab and using default Drive path
    if IS_COLAB and "/content/drive" in user_base_path:
        mount_point = "/content/drive"
        if not os.path.ismount(mount_point):
            print("🔌 Colab detected: Mounting Google Drive...")
            drive.mount(mount_point)
    
    print(f"📂 Working Directory: {user_base_path}")
    return ProjectConfig(user_base_path)


# ==========================================
# 2. CORE LOGIC
# ==========================================

def detect_scenes_cached(
    config: ProjectConfig,
    video_filename: str, 
    threshold: float, 
    min_scene_len: int
) -> List[Dict[str, Any]]:
    """Phase 1: Scene Boundary Detection with Caching."""
    
    paths = config.get_paths(video_filename)
    
    # 1. Check Cache
    if os.path.exists(paths['cuts_file']):
        print(f"   📄 Cache Hit: Loading existing cuts for '{video_filename}'")
        with open(paths['cuts_file'], 'r') as f:
            return json.load(f)
            
    # 2. Validate Input
    if not os.path.exists(paths['video_path']):
        print(f"   ❌ Error: Video not found at {paths['video_path']}")
        return []

    # 3. Run Detection
    print(f"   🎬 Running PySceneDetect (Threshold={threshold})...")
    video = open_video(paths['video_path'])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len))
    
    start_t = time.time()
    scene_manager.detect_scenes(video, show_progress=False) # False to keep CLI clean, can be True
    scene_list = scene_manager.get_scene_list()
    duration = time.time() - start_t
    
    print(f"   ✅ Detected {len(scene_list)} scenes in {duration:.1f}s")

    # 4. Save Cache
    cuts_data = []
    for start, end in scene_list:
        cuts_data.append({
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_seconds": start.get_seconds(),
            "end_seconds": end.get_seconds()
        })
        
    with open(paths['cuts_file'], 'w') as f:
        json.dump(cuts_data, f, indent=2)
        
    return cuts_data


def extract_scenes_resumable(
    config: ProjectConfig,
    video_filename: str, 
    cuts_data: List[Dict[str, Any]]
) -> int:
    """Phase 2: Extraction with Resume capability."""
    
    paths = config.get_paths(video_filename)
    
    # 1. Load Existing Progress
    existing_meta = []
    if os.path.exists(paths['meta_file']):
        try:
            with open(paths['meta_file'], 'r') as f:
                existing_meta = json.load(f)
        except json.JSONDecodeError:
            print("   ⚠️ Corrupted metadata file. Starting fresh.")
    
    finished_ids = {item['scene_id'] for item in existing_meta}
    to_process_count = len(cuts_data) - len(finished_ids)
    
    if to_process_count == 0:
        print("   ⏩ All scenes already extracted. Skipping.")
        return 0

    print(f"   📸 Extracting {to_process_count} new scenes...")
    
    # 2. Processing Loop
    os.makedirs(paths['img_dir'], exist_ok=True)
    cap = cv2.VideoCapture(paths['video_path'])
    new_meta = existing_meta
    
    save_interval = 300 # 5 minutes
    last_save = time.time()
    session_count = 0
    
    for i, cut in enumerate(cuts_data):
        scene_id = f"{paths['video_id']}_scene_{i:04d}"
        
        if scene_id in finished_ids:
            continue
            
        # Extract Middle Frame
        mid_frame = cut['start_frame'] + (cut['end_frame'] - cut['start_frame']) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            img_name = f"{scene_id}.jpg"
            img_path = os.path.join(paths['img_dir'], img_name)
            cv2.imwrite(img_path, frame)
            
            # Record
            new_meta.append({
                "video_id": paths['video_id'],
                "video_name": paths['filename'],
                "scene_id": scene_id,
                "path": img_path,
                "start_time": cut['start_seconds'],
                "end_time": cut['end_seconds'],
                "duration": cut['end_seconds'] - cut['start_seconds']
            })
            session_count += 1
            
        # Checkpoint
        if (time.time() - last_save > save_interval) or (i == len(cuts_data) - 1):
            with open(paths['meta_file'], 'w') as f:
                json.dump(new_meta, f, indent=2)
            last_save = time.time()
            print(f"      💾 Checkpoint: {len(new_meta)} total records saved.")

    cap.release()
    return session_count


def process_video_pipeline(config: ProjectConfig, filename: str, args):
    print(f"\n🎥 Processing: {filename}")
    
    # Phase 1
    cuts = detect_scenes_cached(config, filename, args.threshold, args.min_scene_len)
    if not cuts:
        print("   ⚠️ No cuts found or video missing. Skipping Phase 2.")
        return

    # Phase 2
    processed = extract_scenes_resumable(config, filename, cuts)
    print(f"   🎉 Finished '{filename}'. New scenes extracted: {processed}")


# ==========================================
# 3. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ClipStream Scene Detector")
    
    # Input Sources
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='Single video filename (must be in videos folder)')
    group.add_argument('--video_list', type=str, help='Text file with list of video filenames')
    
    # Config
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream', 
                        help='Root project directory')
    parser.add_argument('--threshold', type=float, default=5.0, help='Scene detection threshold')
    parser.add_argument('--min_scene_len', type=int, default=45, help='Minimum scene length in frames')

    args = parser.parse_args()
    
    # Initialize
    config = setup_environment(args.base_dir)
    
    # Execution
    if args.video:
        process_video_pipeline(config, args.video, args)
        
    elif args.video_list:
        if not os.path.exists(args.video_list):
            print(f"❌ Error: List file '{args.video_list}' not found.")
            sys.exit(1)
            
        with open(args.video_list, 'r') as f:
            videos = [line.strip() for line in f if line.strip()]
            
        print(f"📋 Found {len(videos)} videos in list.")
        for vid in videos:
            process_video_pipeline(config, vid, args)

if __name__ == "__main__":
    main()