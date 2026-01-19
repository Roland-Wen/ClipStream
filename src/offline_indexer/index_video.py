#!/usr/bin/env python3
"""
ClipStream - Master Indexing Pipeline Orchestrator
Author: Roland Wen
Description:
    Coordinates the full end-to-end pipeline:
    1. Scene Detection (Temporal Segmentation)
    2. Embedding Generation (Semantic Mapping)
    3. Pinecone Upload (Cloud Indexing)
    
    Includes benchmarking for pipeline efficiency and time-per-video-hour metrics.
"""

import os
import sys
import time
import argparse
import subprocess
import cv2
from typing import List, Dict, Optional

def get_video_duration(video_path: str) -> float:
    """
    Calculates the total duration of the video in seconds using OpenCV.
    
    Args:
        video_path: Absolute path to the video file.
    Returns:
        Duration in seconds (float). Returns 0.0 if file cannot be opened.
    """
    if not os.path.exists(video_path):
        return 0.0
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration

def run_step(script_path: str, args_list: List[str], desc: str) -> float:
    """
    Executes a sub-pipeline script and measures its execution time.
    
    Args:
        script_path: Path to the .py script to execute.
        args_list: List of CLI arguments for the script.
        desc: A friendly description for logging.
    Returns:
        Elapsed time in seconds (float).
    """
    print(f"\n🚀 PHASE: {desc}")
    print("-" * 40)
    
    start_time = time.time()
    
    # We use sys.executable to ensure we use the same Python environment (e.g., Colab's)
    cmd = [sys.executable, script_path] + args_list
    
    try:
        # check=True will raise an exception if the script fails
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed at Phase: {desc}")
        sys.exit(e.returncode)
    
    elapsed = time.time() - start_time
    print(f"✅ Completed {desc} in {elapsed:.2f}s")
    return elapsed

def print_report(total_time: float, video_duration: float, breakdown: Dict[str, float]):
    """Prints a formatted performance report to the console."""
    print("\n" + "="*50)
    print("📊 CLIPSTREAM PERFORMANCE REPORT")
    print("="*50)
    
    print(f"{'Total Pipeline Time:':<25} {total_time:.2f}s")
    print(f"{'Total Video Duration:':<25} {video_duration:.2f}s")
    
    if video_duration > 0:
        ratio = total_time / video_duration
        speed = 1.0 / ratio
        print(f"{'Processing Speed:':<25} {speed:.2f}x Real-time")
        # Estimate: (sec_taken / sec_video) * 3600 = sec_to_process_1h
        time_per_hour = (total_time / video_duration) * 3600 / 60 # in minutes
        print(f"{'Est. Time per 1h Video:':<25} {time_per_hour:.2f} minutes")
    
    print("\n📉 Breakdown by Phase:")
    for phase, sec in breakdown.items():
        percent = (sec / total_time) * 100
        print(f"  - {phase:<15} {sec:>7.2f}s ({percent:>5.1f}%)")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="ClipStream Master Indexer")
    parser.add_argument('--video', type=str, required=True, help='Filename of video in /videos folder')
    parser.add_argument('--category', type=str, required=True, help='Category (e.g., anime, movie)')
    parser.add_argument('--year', type=int, required=True, help='Release year')
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream', help='Root project dir')
    
    args = parser.parse_args()

    # 1. Setup Paths
    # We assume detection.py, generate_embeddings.py, and upload_vectors.py are in the same folder
    src_dir = os.path.dirname(os.path.abspath(__file__))
    s_detect = os.path.join(src_dir, "detection.py")
    s_embed = os.path.join(src_dir, "generate_embeddings.py")
    s_upload = os.path.join(src_dir, "upload_vectors.py")
    
    video_path = os.path.join(args.base_dir, "videos", args.video)

    # 2. Pre-flight Check
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        sys.exit(1)

    # 3. Execution
    print(f"🎬 Initializing Pipeline for: {args.video}")
    breakdown = {}
    total_start = time.time()

    # STEP 1: DETECTION
    breakdown['Detection'] = run_step(s_detect, [
        "--video", args.video,
        "--base_dir", args.base_dir
    ], "Scene Detection")

    # STEP 2: EMBEDDING
    breakdown['Embedding'] = run_step(s_embed, [
        "--video", args.video,
        "--base_dir", args.base_dir
    ], "CLIP Embedding")

    # STEP 3: UPLOAD
    breakdown['Upload'] = run_step(s_upload, [
        "--video", args.video,
        "--base_dir", args.base_dir,
        "--category", args.category,
        "--year", str(args.year)
    ], "Cloud Upload")

    total_time = time.time() - total_start
    video_duration = get_video_duration(video_path)

    # 4. Report
    print_report(total_time, video_duration, breakdown)

if __name__ == "__main__":
    main()