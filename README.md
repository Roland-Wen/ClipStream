🎬 ClipStream: Semantic Video Search Engine

Project Roadmap & Engineering Plan

🎯 Project Objective

Build a production-grade, end-to-end system that allows users to search through video content (e.g., Baseball games, Anime) using natural language queries (e.g., "A batter hitting a home run").

Target Roles: Machine Learning Engineer, Backend Engineer.
Core Competencies Demonstrated:

ML Ops: Model optimization (ONNX/Quantization), Vector Databases (Pinecone).

Backend: High-performance API (FastAPI), Async architecture.

System Design: Decoupled architecture (Offline Indexing vs. Online Inference).

🏗️ System Architecture

The system is split into two distinct pipelines to maximize free-tier resources and mimic real-world large-scale search systems.

1. Offline Indexing Pipeline (Google Colab - GPU)

Compute: Google Colab Free Tier (Tesla T4 GPU).

Input: Raw Video Files (MP4) from YouTube/Drive.

Process:

Scene Detection: OpenCV splits video into semantic shots.

Embedding: CLIP (Vision Transformer) converts shots to vectors.

Storage: Vectors pushed to Pinecone (Serverless).

Output: Populated Vector Database.

2. Online Inference API (Cloud CPU - Hugging Face/Render)

Compute: CPU-only container (Free Tier).

Input: User text query.

Process:

Text Embedding: ONNX-optimized CLIP Text Encoder (Runs fast on CPU).

Retrieval: Query Pinecone for top K matches.

Response: Return timestamped video links.

🗓️ 10-Week Execution Plan (10 hrs/week)

Phase 1: The Data & ML Pipeline (Weeks 1-3)

Goal: Build the "Indexer" that processes videos and saves them to the database.

Week 1: Video Ingestion & Scene Boundary Detection

Focus: OpenCV, Video Processing, Data Engineering.

Day 1: Environment & Setup

Initialize GitHub repo ClipStream with .gitignore (Python) and README.md.

Set up Google Colab notebook structure (Mount Google Drive).

Write a script to download a specific YouTube video (using yt-dlp) to Drive.

Day 2: Scene Detection Implementation

Option A (Recommended): PySceneDetect

Use the standard library scenedetect which implements robust detect-adaptive algorithms (HSV/luminance based).

Easier to implement than raw OpenCV and handles "fade" transitions better.

Option B (ML Approach): TransNet V2

Use a pre-trained Deep Learning model specifically designed for Shot Boundary Detection.

Higher accuracy but requires more compute/GPU time. Great for showing off "ML Engineering" skills.

Option C (Backend Approach): FFmpeg

Use ffmpeg -vf "select='gt(scene,0.4)',showinfo" to extract timestamps via CLI.

Fastest execution, robust, but less "Pythonic" integration.

Action: Pick one method. (Recommendation: Start with PySceneDetect for speed, upgrade to TransNet V2 if accuracy is low).

Day 3: The "Cutter" Script

Write a function extract_scenes(video_path) that returns list of (start_time, end_time).

Save the middle frame of each scene as a .jpg to a temporary folder.

Verify the quality of cuts manually (do they make sense?).

Day 4: Metadata Structure

Design the JSON metadata schema: {"video_id": str, "start": float, "end": float, "scene_id": str}.

Implement a checkpoint system: Save progress to a .json file every 5 minutes (to survive Colab disconnects).

Test the pipeline on a full 20-minute video file.

Day 5: Code Quality & Refactor

Refactor Colab cells into functions.

Add type hints (def process(path: str) -> List[dict]:).

Milestone: You have a folder of scene images and a JSON catalog of timestamps.

Week 2: The Embedding Engine (CLIP)

Focus: PyTorch, Vision Transformers, Vector Spaces.

Day 1: CLIP Setup

Install transformers and torch in Colab.

Load openai/clip-vit-base-patch32 model on GPU.

Write a test script: Embed one image and one text ("baseball"), calculate cosine similarity.

Day 2: Batch Processing

Implement a DataLoader or simple loop to batch images (Batch size = 32).

Run inference on the images generated in Week 1.

Handle OOM (Out of Memory) errors: Add error catching if batch size is too big.

Day 3: Feature Storage

Save embeddings as .npy (NumPy) files on Google Drive.

Link embeddings to metadata: Create a master list [{"id": "scene_1", "vector": [...], "meta": {...}}].

Validate: Ensure vector dimension is 512.

Day 4: Sanity Check

Write a quick search script inside Colab: Query "Running" against your new .npy files.

Verify visual matches (Does the top result actually show running?).

Tweak Scene Detection threshold if scenes are too short/long.

Day 5: Refinement

Optimize the loop: Ensure GPU utilization is >80%.

Clean up code: Move big logic blocks into a utils.py file you can import in Colab.

Milestone: You have a "Database" of vectors stored as files on Google Drive.

Week 3: Vector Database (Pinecone)

Focus: Cloud Databases, API Integration, Batch Uploads.

Day 1: Pinecone Setup

Sign up for Pinecone Free Tier.

Create an Index: clip-stream (Dimensions: 512, Metric: Cosine).

Install pinecone-client in Colab.

Day 2: Upload Script

Load your .npy data and Metadata JSON from Drive.

Write a function to batch upload (upsert) vectors in chunks of 100.

Add retry logic (exponential backoff) for network failures.

Day 3: Metadata Filtering

Add metadata fields to Pinecone vectors: video_title, season_year.

Test a filtered query in Colab (e.g., "Home run" only in "2024").

Verify data persistence in Pinecone Console.

Day 4: End-to-End Indexing

Run the full pipeline (Video -> Scenes -> CLIP -> Pinecone) on 3-5 different videos.

Measure total time taken per video hour (Benchmark).

Document the "Cost" (Time) of indexing.

Day 5: Review

Milestone: Your "Offline" Indexing pipeline is complete. Data is live in the cloud.

Commit all "Indexer" notebooks/scripts to research/ folder in GitHub.

Phase 2: The Backend Engineering (Weeks 4-7)

Goal: Build the API and Optimize it for CPU inference.

Week 4: API Skeleton (FastAPI)

Focus: Backend Development, REST APIs, Dependency Injection.

Day 1: Project Structure

Create backend/ directory. Structure: app/main.py, app/api/, app/core/.

Initialize poetry or requirements.txt.

Create a basic "Hello World" FastAPI app.

Day 2: Database Connection

Implement get_db_client dependency for Pinecone.

Create a Pydantic model SearchRequest (query: str, top_k: int).

Create a Pydantic model SearchResponse (results: List[VideoMatch]).

Day 3: Search Endpoint V1

Implement POST /search.

Temporary: Load standard PyTorch CLIP model (Text Encoder only) inside the API.

Connect logic: Text -> PyTorch CLIP -> Vector -> Pinecone Query.

Day 4: Local Testing

Test using curl or Postman.

Observe RAM usage (It will be high, likely >1GB).

Observe Latency (It might be slow on CPU).

Day 5: Docker Basics

Write a Dockerfile for the backend.

Test building the image locally.

Milestone: A working (but heavy/slow) Search API.

Week 5: Search Logic & Ranking

Focus: Information Retrieval, Business Logic.

Day 1: Result Ranking

Implement a threshold: Filter out results with similarity score < 0.25.

Format the output: Construct a valid YouTube URL with timestamp (?t=120s).

Day 2: Hybrid Search (Optional)

Add "Keyword Search" capability (if metadata allows).

Allow filtering by video title in the SearchRequest.

Day 3: Async/Await

Ensure all Pinecone network calls are async.

Verify the API is non-blocking (can handle concurrent requests).

Day 4: Error Handling

Add global exception handlers (e.g., Pinecone down, Empty results).

Return proper HTTP 404/500 codes.

Day 5: Pagination

Implement simple pagination (limit/offset) if Pinecone supports it, or just control top_k.

Milestone: A robust API logic layer.

Week 6: ML Optimization (ONNX) - 🌟 CRITICAL

Focus: Model Compression, Latency Reduction, Resume Building.

Day 1: ONNX Export

Write a script to export only the CLIP Text Encoder to ONNX format.

Verify the ONNX model output matches PyTorch output (within tolerance).

Day 2: ONNX Runtime (ORT)

Replace PyTorch in app/services/model.py with onnxruntime.

Implement the Tokenizer + ORT Inference pipeline.

Day 3: Quantization

Apply Dynamic Quantization (Float32 -> Int8) to the ONNX model.

Measure file size reduction (Expect ~4x smaller).

Day 4: Benchmarking

Write a benchmark script: Compare Latency & RAM of PyTorch vs ONNX vs ONNX+Quantized.

Generate a chart for your README/Resume.

Day 5: Integration

Update the FastAPI app to use the quantized.onnx model.

Milestone: API RAM usage drops significantly (fits in Free Tier).

Week 7: Production Engineering

Focus: Logging, Testing, CI/CD.

Day 1: Logging

Replace print with structlog or loguru.

Log structured events: {"event": "search", "query": "baseball", "latency_ms": 120}.

Day 2: Unit Testing

Install pytest.

Write tests for: Input validation, ONNX model loading, Ranking logic.

Day 3: Integration Testing

Mock the Pinecone API (don't hit real DB in tests).

Verify the full search flow returns 200 OK.

Day 4: CI Pipeline

Create .github/workflows/test.yml.

Configure GitHub Actions to run pytest on push.

Day 5: Code Quality

Add ruff or black for linting.

Milestone: A "Professional" codebase ready for scrutiny.

Phase 3: Deployment & Polish (Weeks 8-10)

Goal: Bring it to life and sell it.

Week 8: Frontend (Streamlit)

Focus: UI/UX, Rapid Prototyping.

Day 1: Setup

Create frontend/ directory.

Build a simple Streamlit app with a Text Input box.

Day 2: API Integration

Connect Frontend to Backend: requests.post("http://api.../search").

Display results as a list of Video Players (YouTube embeds).

Day 3: UX Polish

Add a "Jump to Timestamp" button.

Add a "Score" confidence bar for each result.

Day 4: Hosting Config

Create a docker-compose.yml to run Backend + Frontend locally.

Day 5: Demo Prep

Record a screen capture of the local app working perfectly.

Week 9: Deployment

Focus: Cloud Hosting, DevOps.

Day 1: Hugging Face Spaces

Create a new "Space" (Docker SDK).

Configure secrets (Pinecone API Key).

Day 2: Docker Deployment

Push your optimized Docker image to the Space.

Verify the API is live and accessible.

Day 3: Frontend Deploy

Deploy Streamlit to Streamlit Cloud (or same HF Space).

Point Frontend to the live API URL.

Day 4: Monitoring

Check logs in production.

Verify latency is acceptable.

Day 5: Final Bug Fixes

Handle edge cases (Mobile view, special characters in query).

Week 10: The Portfolio Package

Focus: Documentation, Marketing yourself.

Day 1: The README

Write a killer README. Sections: Architecture, Optimization Results, How to Run.

Day 2: The Diagrams

Create an Architecture Diagram (Mermaid/Lucidchart).

Create a "Sequence Diagram" of a search request.

Day 3: The Blog Post (Optional)

Draft a short LinkedIn/Medium post: "How I built a semantic video search engine for $0."

Day 4: Resume Update

Add the project to your resume. Use metrics (e.g., "Reduced inference latency by 60% via ONNX").

Day 5: Submission

Final commit. Tag v1.0.
