import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional, Dict, Any


from schemas import SearchRequest, SearchResponse
from database import PineconeClient, map_metadata_to_match
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
    VERSION: str = "0.2.0"

    # Infrastructure (Keys will be loaded from .env)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "clip-stream"

    # NEW: Default Search Settings
    MIN_SCORE_THRESHOLD: float = 0.26 # Anything below this is likely noise
    
    # Model Config
    MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # Pydantic Settings Config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    """Returns a cached instance of the settings."""
    return Settings()

# --- LIFESPAN (Startup/Shutdown) ---

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    W4D4: Initializes and warms up the CLIP model on startup.
    """
    settings = get_settings()
    logger.info("--- SERVER STARTING ---")
    
    # 1. Initialize & Warm up Encoder
    encoder = CLIPTextEncoder.get_instance(settings.MODEL_NAME)
    encoder.warm_up()
    
    yield # Server is now running and handling requests
    
    logger.info("--- SERVER SHUTTING DOWN ---")


# --- INITIALIZATION ---

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().VERSION,
    lifespan=lifespan
)

# --- DEPENDENCIES ---

def get_db():
    """Dependency that provides the Pinecone index instance."""
    settings = get_settings()
    try:
        return PineconeClient(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

def get_encoder():
    """Dependency to get the already-initialized encoder."""
    settings = get_settings()
    return CLIPTextEncoder.get_instance(settings.MODEL_NAME)

# --- ENDPOINTS ---

@app.get("/health")
async def health_check(db=Depends(get_db)):
    """Health check that also verifies DB connectivity."""
    try:
        stats = db.describe_index_stats()
        return {"status": "healthy", "db_connected": True, "vector_count": stats.total_vector_count}
    except Exception as e:
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

        query_response = db.query(
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

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal search engine error")

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": f"Welcome to the {get_settings().APP_NAME}"}

if __name__ == "__main__":
    import uvicorn
    # In production, we would use a proper command, but this allows for local testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)