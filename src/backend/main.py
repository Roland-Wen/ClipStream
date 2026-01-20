import logging
import time
from fastapi import FastAPI, Depends, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


from schemas import SearchRequest, SearchResponse, VideoMatch
from database import PineconeClient, map_metadata_to_match

# --- CONFIGURATION (W4D1) ---

class Settings(BaseSettings):
    """
    Application settings managed via pydantic-settings.
    Loads variables from a .env file or environment variables.
    """
    # Project Info
    APP_NAME: str = "ClipStream API"
    DEBUG_MODE: bool = True
    VERSION: str = "0.1.0"

    # Infrastructure (Keys will be loaded from .env)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "clip-stream"
    
    # Model Config
    MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # Pydantic Settings Config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    """Returns a cached instance of the settings."""
    return Settings()

# --- INITIALIZATION ---

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().VERSION
)

# --- DATABASE INJECTION ---

def get_db():
    """Dependency that provides the Pinecone index instance."""
    settings = get_settings()
    try:
        return PineconeClient(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
    settings=Depends(get_settings), 
    db=Depends(get_db)
):
    """
    Search endpoint. Currently using a dummy vector for testing DB retrieval.
    """
    start_time = time.time()
    
    # TODO: Placeholder: In W4D4 we will replace this with real CLIP embedding
    dummy_vector = [0.1] * 512 

    try:
        # 1. Query Pinecone
        # We also pass filters if they exist in the request
        filter_dict = {}
        if request.category_filter:
            filter_dict["category"] = {"$eq": request.category_filter}
        if request.year_filter:
            filter_dict["year"] = {"$eq": request.year_filter}

        query_response = db.query(
            vector=dummy_vector,
            top_k=request.top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        # 2. Map Results
        results = [map_metadata_to_match(m) for m in query_response.get("matches", [])]

        return SearchResponse(
            results=results,
            total_matches=len(results),
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