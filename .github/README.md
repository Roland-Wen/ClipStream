🎬 ClipStream

![CI Status](https://github.com/Roland-Wen/ClipStream/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

ClipStream is a semantic video search engine that allows users to query video content using natural language (e.g., "A batter hitting a home run").

It leverages OpenAI's CLIP model for multimodal embedding and Pinecone for vector retrieval, optimized for CPU inference using ONNX Runtime.

🏗️ Architecture

Offline Indexer: Google Colab (T4 GPU) -> OpenCV Scene Detect -> CLIP -> Pinecone.

Online API: FastAPI (CPU Optimized) -> ONNX Quantized Model -> Vector Search.

🚀 Project Roadmap

[x] Week 1: Video Ingestion & Adaptive Scene Detection

[x] Week 2: Feature Extraction with CLIP

[x] Week 3: Vector Database Indexing (Pinecone)

[x] Week 4: FastAPI Backend Development

[x] Week 5: Search Logic & Ranking

[x] Week 6: ONNX Optimization & Quantization

[x] Week 7: Production Engineering (Logging, CI/CD)

[x] Week 8: Streamlit Frontend

[ ] Week 9: Cloud Deployment

[ ] Week 10: Documentation & Release

🛠️ Tech Stack

ML: PyTorch, CLIP, ONNX Runtime

Data: OpenCV, ffmpeg, Pinecone

Backend: FastAPI, Docker

Frontend: Streamlit

👤 Author

Roland Wen Machine Learning & Data Science MS, UCSD
