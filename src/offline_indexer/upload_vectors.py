#!/usr/bin/env python3
"""
ClipStream - Vector Database Uploader (CLI)
Author: Roland Wen
Description: 
    Uploads .npy vectors to Pinecone.
    Injects global metadata (Type, Year) into every record.
"""

import os
import json
import time
import argparse
import sys
import numpy as np

# Conditional imports for environment safety
try:
    from pinecone import Pinecone
    from tqdm import tqdm
    from google.colab import drive, userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# ==========================================
# 1. SETUP
# ==========================================

class ProjectConfig:
    def __init__(self, base_path: str):
        self.base = base_path
        self.embeddings = os.path.join(base_path, "embeddings")
        self.metadata = os.path.join(base_path, "metadata")
        
    def get_paths(self, video_filename: str):
        stem = os.path.splitext(os.path.basename(video_filename))[0]
        return {
            "emb_file": os.path.join(self.embeddings, f"{stem}_embeddings.npy"),
            "ids_file": os.path.join(self.embeddings, f"{stem}_ids.json"),
            "meta_file": os.path.join(self.metadata, f"{stem}_metadata.json")
        }

def get_pinecone_client():
    """Securely retrieves API Key."""
    api_key = None
    if IS_COLAB:
        try:
            api_key = userdata.get('PINECONE_API_KEY')
        except Exception:
            pass
    
    if not api_key:
        api_key = os.getenv('PINECONE_API_KEY')
        
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found in Secrets or Environment.")
        sys.exit(1)
        
    return Pinecone(api_key=api_key)

# ==========================================
# 2. BATCH UPLOAD LOGIC
# ==========================================

def process_upload(config, filename, category, year, index_name="clip-stream"):
    paths = config.get_paths(filename)
    
    # 1. Load Data
    if not os.path.exists(paths['emb_file']):
        print(f"❌ Embeddings missing for {filename}")
        return

    print(f"📦 Loading data for: {filename} ({category}, {year})")
    
    vectors_np = np.load(paths['emb_file'])
    
    with open(paths['ids_file'], 'r') as f:
        ids_list = json.load(f)
        
    with open(paths['meta_file'], 'r') as f:
        meta_list = json.load(f)
        meta_dict = {item['scene_id']: item for item in meta_list}

    # 2. Prepare Payload (With Injection)
    payload = []
    print("   Preparing payload with metadata injection...")
    
    for i, scene_id in enumerate(ids_list):
        vector = vectors_np[i].tolist()
        meta = meta_dict.get(scene_id, {})
        
        # Clean & Inject
        clean_meta = {
            "video_name": meta.get("video_name", filename),
            "start_time": float(meta.get("start_time", 0.0)),
            "end_time": float(meta.get("end_time", 0.0)),
            "path": meta.get("path", ""),
            # INJECTED FIELDS
            "category": category,
            "year": int(year)
        }
        
        payload.append((scene_id, vector, clean_meta))

    # 3. Upload
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    total = len(payload)
    batch_size = 100
    
    print(f"🚀 Uploading {total} vectors to '{index_name}'...")
    
    for i in tqdm(range(0, total, batch_size)):
        chunk = payload[i : i + batch_size]
        
        # Retry loop
        for attempt in range(3):
            try:
                index.upsert(vectors=chunk)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"❌ Failed batch {i}: {e}")
                else:
                    time.sleep(2 ** attempt)

    print("✅ Upload Complete.")

# ==========================================
# 3. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ClipStream Vector Uploader")
    parser.add_argument('--video', type=str, required=True, help='Video filename')
    parser.add_argument('--category', type=str, required=True, help='Category (anime/movie/etc)')
    parser.add_argument('--year', type=int, required=True, help='Year of release')
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream')
    
    args = parser.parse_args()
    
    if IS_COLAB and "/content/drive" in args.base_dir and not os.path.exists("/content/drive"):
         drive.mount("/content/drive")

    config = ProjectConfig(args.base_dir)
    process_upload(config, args.video, args.category, args.year)


if __name__ == "__main__":
    main()