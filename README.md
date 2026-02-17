---
title: ClipStream
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: src/frontend/app.py
pinned: false
---

# 🎬 ClipStream: Semantic Video Search

**ClipStream** is an AI-powered search engine that allows you to find specific moments in videos using natural language. Built with CLIP (Contrastive Language-Image Pre-training) and Pinecone, it bridges the gap between raw video pixels and semantic human descriptions.

### 🚀 High-Level Architecture
- **Frontend:** Streamlit (Hosted here on Hugging Face Spaces).
- **Backend:** FastAPI (Hosted on Render).
- **Vector Database:** Pinecone.
- **Model:** CLIP (ViT-B/32) via Hugging Face Transformers.

---

### 📖 Looking for the Full Documentation?
This branch is optimized for deployment on Hugging Face Spaces. For the full technical breakdown, source code, and offline indexing scripts, visit the **[Main GitHub Repository](https://github.com/roland-wen/clipstream)**.

### 🧪 Features
- **Semantic Search:** Query videos using descriptive phrases (e.g., *"A character sprinting in the rain"*).
- **Interactive Playback:** Jump directly to the identified timestamp in the YouTube source.
- **Hybrid Cloud:** Decoupled architecture for improved security and performance.

---
*Developed by Roland Wen as part of the UCSD MSMLDS Program.*
