import logging
from pinecone import Pinecone
from schemas import VideoMatch
from typing import List

logger = logging.getLogger(__name__)

class PineconeClient:
    """
    Singleton class to manage the Pinecone database connection.
    Ensures only one connection instance exists.
    """
    _instance = None

    def __new__(cls, api_key: str, index_name: str):
        if cls._instance is None:
            logger.info(f"Initializing Pinecone connection for index: {index_name}")
            try:
                pc = Pinecone(api_key=api_key)
                cls._instance = pc.Index(index_name)
                # Verify connection by fetching stats
                cls._instance.describe_index_stats()
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone: {e}")
                raise ConnectionError(f"Could not connect to Pinecone: {e}")
        return cls._instance

def map_metadata_to_match(match: dict) -> VideoMatch:
    """
    Helper to map raw Pinecone metadata to our Pydantic VideoMatch model.
    """
    metadata = match.get("metadata", {})
    return VideoMatch(
        video_id=metadata.get("video_name", "unknown"),
        scene_id=match.get("id"),
        score=round(match.get("score", 0.0), 4),
        start_time=float(metadata.get("start_time", 0.0)),
        end_time=float(metadata.get("end_time", 0.0)),
        video_url=None # Placeholder: URL logic will be added in Week 5
    )