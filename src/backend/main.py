import logging
import time
from fastapi import FastAPI, Depends
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


from schemas import SearchRequest, SearchResponse, VideoMatch

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

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Used for monitoring and verification of server startup.
    """
    settings = get_settings()
    logger.info("Health check requested.")
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "debug_mode": settings.DEBUG_MODE
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, settings: Settings = Depends(get_settings)):
    """
    Search for video scenes using natural language.
    Current Status: MOCK RESPONSE (W4D2)
    """
    start_time = time.time()
    logger.info(f"Received query: '{request.query}' with Top-K: {request.top_k}")

    # TODO: Connect to Pinecone & CLIP (Coming in W4D3-5)
    
    # Mock Data to verify Schema
    mock_results = [
        VideoMatch(
            video_id="anime_op",
            scene_id="anime_op_scene_001",
            score=0.89,
            start_time=12.5,
            end_time=15.0
        )
    ]

    process_time = (time.time() - start_time) * 1000

    return SearchResponse(
        results=mock_results,
        total_matches=len(mock_results),
        processing_time_ms=round(process_time, 2)
    )

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": f"Welcome to the {get_settings().APP_NAME}"}

if __name__ == "__main__":
    import uvicorn
    # In production, we would use a proper command, but this allows for local testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)