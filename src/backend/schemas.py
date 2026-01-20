from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# --- REQUEST MODELS (Input) ---

class SearchRequest(BaseModel):
    """
    Schema for incoming search queries.
    Validation logic ensures queries are safe and meaningful.
    """
    query: str = Field(..., min_length=3, max_length=100, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    
    # Optional Filters (for Hybrid Search later)
    category_filter: Optional[str] = Field(default=None, description="Filter by category (e.g., 'anime')")
    year_filter: Optional[int] = Field(default=None, ge=1900, le=2100, description="Filter by release year")

    @field_validator('query')
    def query_must_not_be_blank(cls, v):
        """Sanitizes input by stripping whitespace."""
        if not v.strip():
            raise ValueError('Query must not be empty strings')
        return v.strip()

# --- RESPONSE MODELS (Output) ---

class VideoMatch(BaseModel):
    """
    Represents a single search result item.
    """
    video_id: str
    scene_id: str
    score: float
    start_time: float
    end_time: float
    video_url: Optional[str] = None # We will construct this later (e.g., YouTube link)
    thumbnail_url: Optional[str] = None # Placeholder for when we have a frontend

class SearchResponse(BaseModel):
    """
    The full JSON response returned to the client.
    """
    results: List[VideoMatch]
    total_matches: int
    processing_time_ms: float