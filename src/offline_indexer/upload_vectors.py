#!/usr/bin/env python3
"""
ClipStream - Vector Database Uploader (CLI)
Author: Roland Wen
Description: 
    Uploads .npy vectors to Pinecone.
    Injects global metadata and resolves Google Drive Direct Links.
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
    from google.colab import drive, auth, userdata
    from googleapiclient.discovery import build
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# ==========================================
# 1. SETUP & DRIVE API
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

def get_drive_service():
    """Authenticates and returns the Google Drive API service."""
    if IS_COLAB:
        auth.authenticate_user()
        return build('drive', 'v3')
    else:
        print("⚠️ Warning: Not in Colab. Drive API search might fail if not authenticated.")
        return None

def get_file_id_cache(service, video_stem):
    """
    OPTIMIZATION: Fetches ALL scene images for this video in one request.
    Returns: Dict { 'filename.jpg': 'drive_file_id' }
    """
    if not service: return {}
    
    print(f"🔎 Caching Drive IDs for scenes matching '{video_stem}'...")
    cache = {}
    page_token = None
    
    while True:
        # Search for files containing the video stem in their name
        # We filter by image/jpeg to avoid picking up the folder itself or json files
        query = f"name contains '{video_stem}' and mimeType = 'image/jpeg' and trashed = false"
        
        try:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                pageSize=1000  # Maximize page size
            ).execute()
            
            for file in results.get('files', []):
                cache[file['name']] = file['id']
                
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        except Exception as e:
            print(f"⚠️ Cache build failed: {e}")
            break
            
    print(f"   ✅ Cached {len(cache)} IDs.")
    return cache


def get_pinecone_client():
    api_key = None
    if IS_COLAB:
        try:
            api_key = userdata.get('PINECONE_API_KEY')
        except Exception:
            pass
    if not api_key:
        api_key = os.getenv('PINECONE_API_KEY')
        
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found.")
        sys.exit(1)
        
    return Pinecone(api_key=api_key)

# ==========================================
# 2. BATCH UPLOAD LOGIC
# ==========================================

def process_upload(config, filename, category, year, yt_id=None, index_name="clip-stream"):
    paths = config.get_paths(filename)
    
    # 1. Load Data
    if not os.path.exists(paths['emb_file']):
        print(f"❌ Embeddings missing for {filename}")
        return

    print(f"📦 Loading data for: {filename}")
    
    vectors_np = np.load(paths['emb_file'])
    with open(paths['ids_file'], 'r') as f:
        ids_list = json.load(f)
    with open(paths['meta_file'], 'r') as f:
        meta_list = json.load(f)
        meta_dict = {item['scene_id']: item for item in meta_list}

    # 2. Initialize Drive Service for Link Generation
    print("🔑 Authenticating with Google Drive API to generate Direct Links...")
    drive_service = get_drive_service()

    # 3. Prepare Payload
    payload = []
    print("   Preparing payload (Resolving Drive IDs)...")
    
    for i, scene_id in tqdm(enumerate(ids_list), total=len(ids_list), desc="Linking"):
        vector = vectors_np[i].tolist()
        meta = meta_dict.get(scene_id, {})
        
        # Original local path (e.g., /content/drive/.../image.jpg)
        local_path = meta.get("path", "")
        image_filename = os.path.basename(local_path)
        
        # Resolve Drive ID
        drive_id_cache = get_file_id_cache(drive_service, paths['video_stem'])
        
        # Construct Direct Link
        # Format: https://drive.google.com/uc?export=view&id=FILE_ID
        drive_id = drive_id_cache.get(image_filename)
        thumb_url = ""
        if drive_id:
            thumb_url = f"https://drive.google.com/uc?export=view&id={drive_id}"
        
        clean_meta = {
            "video_name": meta.get("video_name", filename),
            "start_time": float(meta.get("start_time", 0.0)),
            "end_time": float(meta.get("end_time", 0.0)),
            "category": category,
            "year": int(year),
            "thumbnail_url": thumb_url
        }
        
        # Inject YouTube ID if provided
        if yt_id:
            clean_meta["youtube_id"] = yt_id
            
        payload.append((scene_id, vector, clean_meta))

    # 4. Upload
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    total = len(payload)
    batch_size = 100
    
    print(f"🚀 Uploading {total} vectors to '{index_name}'...")
    
    for i in range(0, total, batch_size):
        chunk = payload[i : i + batch_size]
        for attempt in range(3):
            try:
                index.upsert(vectors=chunk)
                break
            except Exception as e:
                if attempt == 2: print(f"❌ Failed batch {i}: {e}")
                else: time.sleep(1)

    print("✅ Upload Complete.")

# ==========================================
# 3. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ClipStream Vector Uploader")
    parser.add_argument('--video', type=str, required=True, help='Video filename')
    parser.add_argument('--category', type=str, required=True, help='Category')
    parser.add_argument('--year', type=int, required=True, help='Year')
    parser.add_argument('--yt_id', type=str, default=None, help='YouTube ID')
    parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ClipStream')
    
    args = parser.parse_args()
    
    if IS_COLAB and "/content/drive" in args.base_dir and not os.path.exists("/content/drive"):
         drive.mount("/content/drive")

    config = ProjectConfig(args.base_dir)
    process_upload(config, args.video, args.category, args.year, args.yt_id)

if __name__ == "__main__":
    main()