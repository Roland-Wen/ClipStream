import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Dict, Any, Optional

from schemas import SearchRequest, SearchResponse
from database import PineconeAsyncClient, map_metadata_to_match
from encoder import CLIPTextEncoder

# --- CONFIGURATION  ---

class Settings(BaseSettings):
    """
    Application settings managed via pydantic-settings.
    Loads variables from a .env file or environment variables.
    """
    # Project Info
    APP_NAME: str = "ClipStream API"
    DEBUG_MODE: bool = True
    VERSION: str = "0.3.0"

    # Infrastructure (Keys will be loaded from .env)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "clip-stream"

    # Default Search Settings
    MIN_SCORE_THRESHOLD: float = 0.26 # Anything below this is likely noise

    # Control which engine powers the search
    # Options: "torch" (Baseline), "onnx" (Optimized), "onnx_quant" (Extreme Speed)
    MODEL_BACKEND: str = "onnx"

    # Explicitly list the exact protocol, domain, and port of frontend
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501"   # Alternative local address
    ]
    
    # Model Config
    # MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # Pydantic Settings Config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    """Returns a cached instance of the settings."""
    return Settings()


# --- UTILS: PINECONE FILTER BUILDER ---
def build_pinecone_filters(request: SearchRequest) -> Optional[Dict[str, Any]]:
    """
    Constructs a Pinecone-compatible filter dictionary.
    Supports multi-select via the $in operator.
    """
    filters = {}

    # If user selected specific categories: { "category": { "$in": ["anime", "sports"] } }
    if request.categories and len(request.categories) > 0:
        filters["category"] = {"$in": request.categories}

    # If user selected specific years: { "year": { "$in": [2021, 2024] } }
    if request.years and len(request.years) > 0:
        filters["year"] = {"$in": [int(y) for y in request.years]}

    return filters if filters else None

# --- LIFESPAN (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("--- SERVER STARTING (ONNX PURE) ---")
    
    # 1. Warm up AI (No arguments needed now)
    CLIPTextEncoder.get_instance().warm_up()
    
    # 2. Init Pinecone
    await PineconeAsyncClient.get_index(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
    
    yield
    
    logger.info("--- SERVER SHUTTING DOWN ---")
    await PineconeAsyncClient.close()


# --- INITIALIZATION ---

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ORIGINS,  # Allows specific origins (or * for all)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- GLOBAL ERROR HANDLING ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles expected errors (like 404 Not Found) with a clean JSON format.
    """
    logger.warning(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles unexpected crashes (500 Internal Server Error).
    """
    logger.error(f"Global Crash: {exc}", exc_info=True) # exc_info gives us the stack trace in logs
    return JSONResponse(
        status_code=500,
        content={
            "status": "critical_error", 
            "message": "An internal server error occurred. Please contact support."
        },
    )

# --- DEPENDENCIES ---

async def get_db():
    """Dependency that provides the Async Pinecone index instance."""
    settings = get_settings()
    try:
        return await PineconeAsyncClient.get_index(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

def get_encoder():
    """Dependency to get the already-initialized encoder."""
    settings = get_settings()
    return CLIPTextEncoder.get_instance()

# --- ENDPOINTS ---

@app.get("/health")
async def health_check(db=Depends(get_db)):
    """Health check that also verifies DB connectivity."""
    try:
        stats = await db.describe_index_stats()
        return {"status": "healthy", "db_connected": True, "vector_count": stats.total_vector_count}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest, 
    sort_by: str = Query("confidence", enum=["confidence", "time"]), # Added sort toggle
    db=Depends(get_db),
    encoder: CLIPTextEncoder = Depends(get_encoder),
    settings: Settings = Depends(get_settings)
):
    """
    Search endpoint with Thresholding and Sorting logic 
    """
    start_time = time.time()
    
    try:
        # 1. Generate real embedding from user query
        query_vector = encoder.encode_text(request.query)

        # 2. Build Hybrid Filters
        pinecone_filters = build_pinecone_filters(request)

        query_response = await db.query(
            vector=query_vector,
            top_k=request.top_k,
            include_metadata=True,
            filter=pinecone_filters
        )

        # 3. Map and Filter by Threshold
        all_results = [map_metadata_to_match(m) for m in query_response.get("matches", [])]
        
        # Calculate MAX SCORE for logs (before we filter or sort)
        # This tells us the absolute best match found by the AI
        max_score = max([res.score for res in all_results]) if all_results else 0.0

        filtered_results = [
            res for res in all_results 
            if res.score >= settings.MIN_SCORE_THRESHOLD
        ]

        # ---No results handling---
        if not filtered_results:
            # Distinguish between "Pinecone found nothing" vs "Threshold filtered everything"
            if not all_results:
                msg = "No video content found matching your query criteria."
            else:
                msg = (f"Found {len(all_results)} matches, but they were all below the "
                       f"confidence threshold ({settings.MIN_SCORE_THRESHOLD}). Try a more specific query.")
            
            # We use 404 to indicate 'Resource Not Found'
            raise HTTPException(status_code=404, detail=msg)

        # 4. Sort
        if sort_by == "time":
            filtered_results.sort(key=lambda x: x.start_time)
        else:
            filtered_results.sort(key=lambda x: x.score, reverse=True)

        # 5. Logging
        # We log max_score to see if the AI found ANYTHING, 
        # even if it was below our display threshold.
        logger.info(
            f"Query: [{request.query}] | "
            f"Max Score: {max_score:.4f} | "
            f"Returned: {len(filtered_results)}/{len(all_results)} | "
            f"Sort: {sort_by}"
        )
        return SearchResponse(
            results=filtered_results,
            total_matches=len(filtered_results),
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions (like our new 404) so they pass through
        raise he
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": f"Welcome to the {get_settings().APP_NAME}"}

if __name__ == "__main__":
    import uvicorn
    # In production, we would use a proper command, but this allows for local testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)