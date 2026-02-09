#!/usr/bin/env python3
"""
ClipStream - Local Search Verification (CLI)
Author: Roland Wen
Description: 
    Queries generated .npy files to verify vector quality.
    Does NOT require a Vector DB (runs in-memory).

Usage:
    python test_search.py --video "episode_01.mp4" --query "a cat eating"
"""

import os
import json
import argparse
import sys
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 1. SETUP
# ==========================================

class ProjectConfig:
    def __init__(self, base_path: str):
        self.embeddings = os.path.join(base_path, "embeddings")
        self.metadata = os.path.join(base_path, "metadata")
        
    def get_paths(self, video_filename: str):
        filename = os.path.basename(video_filename)
        stem = os.path.splitext(filename)[0]
        return {
            "emb_file": os.path.join(self.embeddings, f"{stem}_embeddings.npy"),
            "ids_file": os.path.join(self.embeddings, f"{stem}_ids.json"),
            "meta_file": os.path.join(self.metadata, f"{stem}_metadata.json")
        }

# ==========================================
# 2. SEARCH LOGIC
# ==========================================

def run_search(config, filename, query_text, top_k=5):
    paths = config.get_paths(filename)
    
    # Validate Files
    if not os.path.exists(paths['emb_file']):
        print(f"❌ Embeddings not found: {paths['emb_file']}")
        return

    # Load Data
    print(f"⏳ Loading vectors for '{filename}'...")
    video_vectors = np.load(paths['emb_file'])
    
    with open(paths['ids_file'], 'r') as f:
        scene_ids = json.load(f)
        
    with open(paths['meta_file'], 'r') as f:
        meta_dict = {item['scene_id']: item for item in json.load(f)}

    # Load Model (Text Only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                              revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    
    # Encode Query
    print(f"🔍 Searching for: '{query_text}'")
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Dot Product
    video_tensor = torch.tensor(video_vectors).to(device)
    similarity = (text_features @ video_tensor.T).squeeze(0)
    
    # Rank
    values, indices = similarity.topk(top_k)
    
    print(f"\n🏆 Top {top_k} Matches:")
    print("-" * 50)
    
    for score, idx in zip(values, indices):
        idx = idx.item()
        score = score.item()
        
        scene_id = scene_ids[idx]
        scene_info = meta_dict.get(scene_id)
        start_time = scene_info['start_time']
        end_time = scene_info['end_time']
        timestamp = f"{int(start_time//60)}:{int(start_time%60)} - {int(end_time//60)}:{int(end_time%60)}"
        
        print(f"   Confidence: {score*100:.2f}% | Time: {timestamp} | ID: {scene_id}")

# ==========================================
# 3. ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ClipStream Local Search Test")
    parser.add_argument('--video', type=str, required=True, help='Video filename to search')
    parser.add_argument('--query', type=str, required=True, help='Text query')
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream')
    
    args = parser.parse_args()
    
    # Auto-mount if needed
    try:
        from google.colab import drive
        if "/content/drive" in args.base_dir and not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    except ImportError:
        pass

    config = ProjectConfig(args.base_dir)
    run_search(config, args.video, args.query)

if __name__ == "__main__":
    main()
