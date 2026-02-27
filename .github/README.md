# ClipStream 🎬 

[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/w80707/ClipStream)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CI Status](https://github.com/Roland-Wen/ClipStream/actions/workflows/ci.yml/badge.svg)
[![Docker Image](https://img.shields.io/badge/Docker-GHCR-blue)]([https://github.com/roland-wen/clipstream/pkgs/container/clipstream-api](https://github.com/users/Roland-Wen/packages/container/package/clipstream-api))

**ClipStream** is a semantic video search engine that allows you to search through video content using natural language queries. Instead of relying on manual tags, ClipStream uses **CLIP (Contrastive Language-Image Pre-training)** to jump directly to the most relevant timestamps in a video.

![clipStreamDemo](https://github.com/user-attachments/assets/6e4bd963-6555-417d-837c-f3b39cb9e74c)

---

## 🚀 Quick Start (Production Image)

Run the fully optimized, non-root production backend directly from GitHub Container Registry (GHCR):

```bash
docker run -p 8000:8000 \
  -e PINECONE_API_KEY="your_api_key" \
  -e PINECONE_INDEX_NAME="clip-stream" \
  -e LOGTAIL_TOKEN="your_better_stack_token" \
  ghcr.io/roland-wen/clipstream-api:latest
```

## 🏗️ System Architecture

ClipStream is built with a decoupled, "hardware-aware" architecture designed to perform on resource-constrained environments like Render's 512MB free tier.

**Key Technical Decisions**
 - **ONNX Optimization**: The CLIP Text Encoder is converted to ONNX (Open Neural Network Exchange) and optimized for CPU inference, staying within strict RAM limits while maintaining sub-4s latency.
 - **Vector Search**: Utilizes Pinecone for high-dimensional similarity search, allowing for $O(\log N)$ retrieval of video embeddings.
 - **Production Hardening**: The API runs as a non-root user (`clipuser`) inside Docker for security and uses Gunicorn to manage the FastAPI process.
 - **Observability**: Integrated with Loguru and Better Stack (Logtail) for centralized JSON logging and real-time latency monitoring.

 ## 🛠️ Installation & Local Development
 **Prerequisites**
 - Python 3.11+
 - Docker (Optional, for containerized dev)
 - Pinecone API Key

 **Local Setup**
1. **Clone the repository:**
```bash
git clone https://github.com/roland-wen/clipstream.git
cd clipstream/src/backend
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Configure Environment:**
Create a `.env` file in `src/backend/`:
```bash
PINECONE_API_KEY=your_key
DEBUG_MODE=True
JSON_LOGS=False
```
4. **Run the Backend:**
```bash
python main.py
```

## 📉 Troubleshooting
| Issue                             | Solution                                                                                      |
|-----------------------------------|-----------------------------------------------------------------------------------------------|
| **OOM / 503 Errors**              | The model requires ~450MB of RAM. Ensure your host has at least 512MB of total system memory. |
| **Rate Limiting (429)**           | The production API is limited to 20 requests per minute to ensure stability.                  |
| **ImportError: 'LogtailHandler'** | Ensure you have `logtail-python` installed, NOT the legacy `logtail` package.                 |

## 🛣️ Roadmap
- [ ] Real-time Indexing: Build a serverless worker to index YouTube URLs on the fly.

- [ ] Semantic Caching: Implement Redis to cache frequent query embeddings.

## 👨‍💻 Author
**Roland Wen** Graduate Student at UCSD | Machine Learning & Data Science
