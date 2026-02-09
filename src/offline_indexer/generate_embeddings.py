#!/usr/bin/env python3
"""
ClipStream - Vector Embedding Generator (CLI)
Author: Roland Wen
Description: 
    Converts extracted scene images into CLIP vectors.
    Saves output as compressed .npy matrices and .json ID maps.

Usage:
    python generate_embeddings.py --video "episode_01.mp4"
    python generate_embeddings.py --video_list "queue.txt"
"""

import os
import json
import argparse
import sys
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any

# Conditional imports
try:
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPProcessor, CLIPModel
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    # If running locally without these libs, this script will fail gracefully later

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

class ProjectConfig:
    def __init__(self, base_path: str):
        self.base = base_path
        self.scenes = os.path.join(base_path, "scenes")
        self.metadata = os.path.join(base_path, "metadata")
        self.embeddings = os.path.join(base_path, "embeddings")
        
        os.makedirs(self.embeddings, exist_ok=True)

    def get_paths(self, video_filename: str) -> Dict[str, str]:
        filename = os.path.basename(video_filename)
        stem = os.path.splitext(filename)[0]
        return {
            "meta_file": os.path.join(self.metadata, f"{stem}_metadata.json"),
            "emb_file": os.path.join(self.embeddings, f"{stem}_embeddings.npy"),
            "ids_file": os.path.join(self.embeddings, f"{stem}_ids.json"),
            "video_stem": stem
        }

def setup_environment(user_base_path: str) -> ProjectConfig:
    if IS_COLAB and "/content/drive" in user_base_path:
        mount_point = "/content/drive"
        if not os.path.ismount(mount_point):
            print("🔌 Colab detected: Mounting Google Drive...")
            drive.mount(mount_point)
    return ProjectConfig(user_base_path)

# ==========================================
# 2. DATASET CLASS
# ==========================================

class SceneDataset(Dataset):
    def __init__(self, metadata_list: List[Dict], processor: Any):
        self.metadata = metadata_list
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = item['path']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Return black image on failure to prevent crash
            print(f"⚠️ Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # Preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "scene_id": item['scene_id']
        }

# ==========================================
# 3. CORE LOGIC
# ==========================================

def process_video_embeddings(config: ProjectConfig, filename: str, model: Any, processor: Any, device: str):
    paths = config.get_paths(filename)
    
    # Check if metadata exists
    if not os.path.exists(paths['meta_file']):
        print(f"❌ Metadata missing for '{filename}'. Run detection.py first.")
        return

    # Check if already done
    if os.path.exists(paths['emb_file']) and os.path.exists(paths['ids_file']):
        print(f"⏩ Embeddings already exist for '{filename}'. Skipping.")
        return

    print(f"📊 Processing: {filename}")
    
    with open(paths['meta_file'], 'r') as f:
        metadata = json.load(f)
        
    if not metadata:
        print("   ⚠️ No scenes found in metadata.")
        return

    # Create DataLoader
    dataset = SceneDataset(metadata, processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    all_embeddings = []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="   Computing Vectors", unit="batch"):
            pixel_values = batch['pixel_values'].to(device)
            ids = batch['scene_id']
            
            # Forward Pass
            features = model.get_image_features(pixel_values=pixel_values)
            # Normalize
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(features.cpu().numpy())
            all_ids.extend(ids)

    # Save
    final_embeddings = np.vstack(all_embeddings)
    np.save(paths['emb_file'], final_embeddings)
    
    with open(paths['ids_file'], 'w') as f:
        json.dump(all_ids, f)
        
    print(f"   ✅ Saved: {final_embeddings.shape} matrix to {os.path.basename(paths['emb_file'])}")


# ==========================================
# 4. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ClipStream Embedding Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='Single video filename')
    group.add_argument('--video_list', type=str, help='Text file with list of filenames')
    
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream', help='Project root')
    args = parser.parse_args()
    
    config = setup_environment(args.base_dir)
    
    # Load Model Once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device: {device.upper()}")
    if device == "cpu":
        print("⚠️ Warning: Running on CPU is slow.")
    
    print("🔄 Loading CLIP Model...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                          revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                  revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Execution Loop
    if args.video:
        process_video_embeddings(config, args.video, model, processor, device)
        
    elif args.video_list:
        if not os.path.exists(args.video_list):
            print(f"❌ List file '{args.video_list}' not found.")
            sys.exit(1)
            
        with open(args.video_list, 'r') as f:
            videos = [line.strip() for line in f if line.strip()]
            
        print(f"📋 Processing batch of {len(videos)} videos...")
        for vid in videos:
            process_video_embeddings(config, vid, model, processor, device)

if __name__ == "__main__":
    main()