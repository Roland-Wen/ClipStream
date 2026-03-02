# ClipStream Technical Documentation 🛠️
This document outlines the internal workflows, data structures, and build processes for the ClipStream semantic search engine.

## 1. System Pipelines

### A. Search Pipeline (Live)

This flow describes how a user query is transformed into a timestamped video result.

```mermaid
graph TD
    User((User)) -->|Search Query| FE[Streamlit Frontend]
    FE -->|POST /search| BE[FastAPI Backend]
    
    subgraph "Inference Engine"
        BE -->|Text String| ONNX[CLIP Text Encoder - ONNX]
        ONNX -->|512D Vector| VEC((Embedding))
    end
    
    subgraph "Vector Database"
        VEC -->|Similarity Search| PC[(Pinecone Index)]
        PC -->|Matches + Metadata| BE
    end
    
    BE -->|Filter & Sort| BE
    BE -->|JSON Response| FE
    FE -->|Embed Video| User
```

### B. Ingestion Pipeline (Offline/Batch)

This describes how raw video files are processed to populate the Pinecone index.

```mermaid
graph LR
    Vid[(Raw Video)] -->|OpenCV| Frames[Scene Extraction]
    Frames -->|Sample 1 frame/scene| CLIP[CLIP Image Encoder]
    CLIP -->|Image Embeddings| Embeds((Vectors))
    
    Embeds -->|Attach Metadata| Metadata[Timestamp/Category/...]
    Metadata -->|Upsert| PC[(Pinecone)]
```

## 2. Pinecone Metadata Schema

ClipStream relies on structured metadata attached to each vector to enable filtering and precise video seeking.

|   **Field**   | **Type** |                      **Description**                     |         **Example**         |
|:-------------:|:--------:|:--------------------------------------------------------:|:---------------------------:|
| video_name    | string   | The title or identifier of the source video.             | "attack_on_titan_s1"        |
| start_time    | float    | The exact timestamp (in seconds) where the match begins. | 124.5                       |
| end_time      | float    | The exact timestamp (in seconds) where the match ends.   | 130.5                       |
| category      | string   | Used for multi-select filtering in the UI.               | "anime", "amv"              |
| year          | integer  | Allows for temporal filtering of content.                | 2024                        |
| thumbnail_url | string   | URL to the representative frame for the search result.   | https://img.youtube.com/... |

Filtering Logic: The backend constructs a Pinecone filter using the $in operator for category and year to ensure sub-millisecond retrieval of context-aware results.

## 3. Multi-Stage Docker Build Process

To minimize the production footprint and enhance security, ClipStream utilizes a two-stage build.

```mermaid
graph TD
    subgraph "Stage 1: Builder (Heavy)"
        B1[python:3.11-slim] --> B2[Install build-essential]
        B2 --> B3[Install requirements.txt]
        B3 --> B4[Create VirtualEnv]
    end
    
    subgraph "Stage 2: Runtime (Light)"
        R1[python:3.11-slim] --> R2[Create 'clipuser' Non-Root]
        B4 -->|Copy venv| R3[Optimized Environment]
        R3 --> R4[Copy ONNX & Source Code]
        R4 --> R5[Set USER clipuser]
    end
    
    R5 --> CMD[Gunicorn Start]
```

**Build Advantages**

1. **Size Reduction**: The final image excludes gcc and build tools, saving ~300MB.

2. **Security**: By using USER clipuser, even if the FastAPI app is compromised, the attacker has no root privileges within the container.

3. **Reproducibility**: The virtual environment is locked during Stage 1, ensuring no dependency drift in production.
