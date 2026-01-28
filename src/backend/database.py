import logging
import asyncio
from pinecone import PineconeAsyncio
from schemas import VideoMatch
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class PineconeAsyncClient:
    """
    Singleton class managing the official Pinecone Async client.
    """
    _client_instance: Optional[PineconeAsyncio] = None
    _index_instance = None
    _lock = asyncio.Lock() # Prevents double-initialization

    @classmethod
    async def get_index(cls, api_key: str, index_name: str):
        # Double-checked locking pattern
        if cls._index_instance is None:
            async with cls._lock:
                # Check again inside the lock
                if cls._index_instance is None:
                    logger.info(f"🚀 Initializing Pinecone Async client: {index_name}")
                    try:
                        cls._client_instance = PineconeAsyncio(api_key=api_key)
                        
                        # Get host first
                        description = await cls._client_instance.describe_index(index_name)
                        index_host = description.host
                        
                        # Use the correct IndexAsyncio method
                        cls._index_instance = cls._client_instance.IndexAsyncio(
                            name=index_name, 
                            host=index_host
                        )
                        
                        await cls._index_instance.describe_index_stats()
                        logger.info(f"✅ Connected to host: {index_host}")
                    except Exception as e:
                        logger.error(f"❌ Failed to connect: {e}")
                        if cls._client_instance:
                            await cls._client_instance.close()
                        cls._index_instance = None
                        cls._client_instance = None
                        raise ConnectionError(f"Could not connect to Pinecone: {e}")
                        
        return cls._index_instance

    @classmethod
    async def close(cls):
        """Closes the session during shutdown."""
        async with cls._lock:
            # 1. Attempt to close the Index connection explicitly
            # (The data plane might have its own session independent of the control plane)
            if cls._index_instance:
                try:
                    # Check if the index instance has a close method and await it
                    if hasattr(cls._index_instance, "close"):
                        logger.info("🔌 Closing Pinecone Index connection...")
                        await cls._index_instance.close()
                except Exception as e:
                    # Ignore errors here, we just want to try our best to clean up
                    logger.warning(f"⚠️ Could not close index instance: {e}")
                finally:
                    cls._index_instance = None

            # 2. Close the main Pinecone Client (Control Plane)
            if cls._client_instance:
                logger.info("🔌 Closing Pinecone Async client session...")
                await cls._client_instance.close()
                cls._client_instance = None

            # 3. Final Yield: Give aiohttp time to close the underlying sockets
            # We sleep AFTER setting variables to None to ensure references are dropped
            await asyncio.sleep(0.5)


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