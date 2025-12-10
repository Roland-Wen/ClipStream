🎬 ClipStream

ClipStream is a semantic video search engine that allows users to query video content using natural language (e.g., "A batter hitting a home run").

It leverages OpenAI's CLIP model for multimodal embedding and Pinecone for vector retrieval, optimized for CPU inference using ONNX Runtime.

🏗️ Architecture

Offline Indexer: Google Colab (T4 GPU) -> OpenCV Scene Detect -> CLIP -> Pinecone.

Online API: FastAPI (CPU Optimized) -> ONNX Quantized Model -> Vector Search.

🚀 Project Roadmap

[ ] Week 1: Video Ingestion & Adaptive Scene Detection

[ ] Week 2: Feature Extraction with ViT (Vision Transformer)

[ ] Week 3: Vector Database Indexing (Pinecone)

[ ] Week 4: FastAPI Backend Development

[ ] Week 5: Search Logic & Ranking

[ ] Week 6: ONNX Optimization & Quantization (Critical Path)

[ ] Week 7: Production Engineering (Logging, CI/CD)

[ ] Week 8: Streamlit Frontend

[ ] Week 9: Cloud Deployment

[ ] Week 10: Documentation & Release

🛠️ Tech Stack

ML: PyTorch, CLIP, ONNX Runtime

Data: OpenCV, ffmpeg, Pinecone

Backend: FastAPI, Docker

Frontend: Streamlit

👤 Author

Roland Wen Machine Learning & Data Science MS, UCSD
