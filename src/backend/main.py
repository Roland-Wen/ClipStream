import sys
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Dict, Any, Optional
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded



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
    VERSION: str = "0.5.0"

    # Infrastructure (Keys will be loaded from .env)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "clip-stream"

    # Default Search Settings
    MIN_SCORE_THRESHOLD: float = 0.26 # Anything below this is likely noise

    # Explicitly list the exact protocol, domain, and port of frontend
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",  # Alternative local address
        "https://w80707-clipstream.hf.space"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True  # Enable for production

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
    setup_logging(settings)
    
    logger.info("--- SERVER STARTING ---", version=settings.VERSION, mode="ONNX Pure")
    
    # 1. Warm up AI (No arguments needed now)
    CLIPTextEncoder.get_instance().warm_up()
    
    # 2. Init Pinecone
    await PineconeAsyncClient.get_index(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
    
    yield
    
    logger.info("--- SERVER SHUTTING DOWN ---")
    await PineconeAsyncClient.close()

# --- LOGGING SETUP ---
def setup_logging(settings: Settings):
    """Configures loguru with a safe request_id default."""
    logger.remove()

    # This 'patch' ensures 'request_id' always exists in 'extra'
    # so the formatter doesn't crash on startup/shutdown logs.
    def patch_record(record):
        if "request_id" not in record["extra"]:
            record["extra"]["request_id"] = "SYSTEM"

    logger.configure(patcher=patch_record)

    if settings.JSON_LOGS:
        logger.add(sys.stdout, serialize=True, level=settings.LOG_LEVEL)
    else:
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[request_id]}</cyan> - <level>{message}</level> <dim>{extra}</dim>",
            level=settings.LOG_LEVEL
        )

# --- MIDDLEWARE ---
async def request_id_middleware(request: Request, call_next):
    """
    Generates a unique Request ID for every incoming call.
    Binds it to the logger context so all subsequent logs include it.
    """
    request_id = str(uuid.uuid4())
    
    # Contextualize: All logs inside this block get 'request_id' automatically
    with logger.contextualize(request_id=request_id):
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            process_time = (time.perf_counter() - start_time) * 1000
            logger.info("Request completed", status_code=response.status_code, duration_ms=round(process_time, 2))
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise e

# --- INITIALIZATION ---
app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().VERSION,
    lifespan=lifespan
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.middleware("http")(request_id_middleware)
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
    return CLIPTextEncoder.get_instance()

# --- ENDPOINTS ---

@app.get("/health")
@limiter.exempt
async def health_check(db=Depends(get_db)):
    """Health check that also verifies DB connectivity."""
    try:
        stats = await db.describe_index_stats()
        return {"status": "healthy", "db_connected": True, "vector_count": stats.total_vector_count}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search(
    request: Request,
    search_req: SearchRequest, # Renamed to avoid collision with 'request'
    sort_by: str = Query("confidence", enum=["confidence", "time"]),
    db=Depends(get_db),
    encoder: CLIPTextEncoder = Depends(get_encoder),
    settings: Settings = Depends(get_settings)
):
    """
    Search endpoint with Thresholding and Sorting logic 
    """
    total_start = time.perf_counter()    
    metrics = {}

    try:
        # PHASE 1: Embedding (CPU Bound)
        t0 = time.perf_counter()
        query_vector = encoder.encode_text(search_req.query)
        metrics["latency_embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug("Embedding generated", latency=metrics["latency_embedding_ms"])
        
        # PHASE 2: Retrieval (IO Bound)
        pinecone_filters = build_pinecone_filters(search_req)
        
        t1 = time.perf_counter()
        query_response = await db.query(
            vector=query_vector,
            top_k=search_req.top_k,
            include_metadata=True,
            filter=pinecone_filters
        )
        metrics["latency_pinecone_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        logger.debug("Vectors retrieved", latency=metrics["latency_pinecone_ms"], matches=len(query_response.get("matches", [])))

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
        total_time = round((time.perf_counter() - total_start) * 1000, 2)
        logger.info(
            "Search executed successfully",
            query=search_req.query,
            results_count=len(filtered_results),
            max_score=round(max_score, 4),
            total_latency_ms=total_time,
            breakdown=metrics
        )
        return SearchResponse(
            results=filtered_results,
            total_matches=len(filtered_results),
            processing_time_ms=total_time
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions (like our new 404) so they pass through
        raise he
    except Exception as e:
        logger.exception("Search endpoint crashed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": f"Welcome to the {get_settings().APP_NAME}"}

if __name__ == "__main__":
    import uvicorn
    # In production, we would use a proper command, but this allows for local testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=None) # nosec B104