from loguru import logger
import asyncio
from pinecone import PineconeAsyncio
from schemas import VideoMatch
from typing import Optional


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
    start_time = float(metadata.get("start_time", 0.0))
    yt_id = metadata.get("youtube_id")
    
    # Construct the video URL if a YouTube ID exists
    # If no YT ID, fallback to the original path (though it won't play in browser)
    video_url = metadata.get("path")
    if yt_id:
        video_url = f"https://www.youtube.com/watch?v={yt_id}&t={int(start_time)}s"

    # Replace the old logic in map_metadata_to_match:
    thumbnail_url = metadata.get("thumbnail_url")
    if thumbnail_url and "drive.google.com" in thumbnail_url:
        # Extract the ID from the existing link
        drive_id = thumbnail_url.split("id=")[-1]
        # New Format: High-res thumbnail endpoint (up to 1000px)
        thumbnail_url = f"https://lh3.googleusercontent.com/u/0/d/{drive_id}=w1000-h1000"

    return VideoMatch(
        video_id=metadata.get("video_name", "unknown"),
        scene_id=match.get("id"),
        score=round(match.get("score", 0.0), 4),
        start_time=start_time,
        end_time=float(metadata.get("end_time", 0.0)),
        thumbnail_url=thumbnail_url,
        video_url=video_url
    )