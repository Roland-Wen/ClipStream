from unittest.mock import AsyncMock, MagicMock

# --- HAPPY PATH TESTS ---

def test_health_check(client, mock_pinecone_db):
    """Verify /health returns 200 and DB stats."""

    # Create the result object we want AFTER the await
    stats_result = MagicMock(total_vector_count=999) # Use distinct number to verify override

    # Configure the AsyncMock to return this object when awaited
    mock_pinecone_db.describe_index_stats = AsyncMock(return_value=stats_result)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["vector_count"] == 999

def test_search_successful(client, mock_pinecone_db):
    """Verify a valid search returns results and correct fields."""

    # 1. Setup Mock Data (A high score match)
    mock_pinecone_db.query = AsyncMock(return_value={
        "matches": [
            {
                "id": "scene_001",
                "score": 0.85,
                "metadata": {
                    "video_name": "test_video.mp4",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "path": "/data/img.jpg",
                    "category": "anime",
                    "year": 2024
                }
            }
        ]
    })

    # 2. Execute Request
    payload = {"query": "test query", "top_k": 5}
    response = client.post("/search", json=payload)

    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["score"] == 0.85
    assert data["results"][0]["video_id"] == "test_video.mp4"

# --- EDGE CASE TESTS ---

def test_search_empty_query_validation(client):
    """Test that empty strings trigger 422 Validation Error (Pydantic)."""
    payload = {"query": "   ", "top_k": 5}
    response = client.post("/search", json=payload)
    
    # Should fail fast at schema validation
    assert response.status_code == 422
    assert "Query must not be empty strings" in response.text

def test_search_non_ascii_query(client, mock_pinecone_db, mock_encoder):
    """Test that Japanese/Non-ASCII characters are handled correctly."""
    # Ensure encoder receives the raw string
    payload = {"query": "猫が走っている", "top_k": 1}

    # Mock empty result is fine, we just want to ensure no crash
    mock_pinecone_db.query = AsyncMock(return_value={"matches": []})

    response = client.post("/search", json=payload)

    # Verify encoder was called with the Japanese text
    mock_encoder.encode_text.assert_called_with("猫が走っている")

    # Since we mocked empty results, we expect 404 (No matches found)
    assert response.status_code == 404
    assert "No video content found" in response.json()["message"]

def test_search_threshold_filtering(client, mock_pinecone_db):
    """Test that results below MIN_SCORE_THRESHOLD are filtered out."""

    # Mock matches that are TOO LOW (0.1 < 0.26)
    mock_pinecone_db.query = AsyncMock(return_value={
        "matches": [
            {"id": "bad_match", "score": 0.10, "metadata": {}}
        ]
    })

    payload = {"query": "irrelevant query", "top_k": 5}
    response = client.post("/search", json=payload)

    # Expect 404 because all results were filtered
    assert response.status_code == 404
    # Verify the specific error message logic
    assert "below the confidence threshold" in response.json()["message"]

# --- ERROR HANDLING TESTS ---

def test_pinecone_failure(client, mock_pinecone_db):
    """Test that DB connection errors return 500 cleanly."""

    # Force the DB mock to raise an exception
    mock_pinecone_db.query = AsyncMock(side_effect=Exception("Connection Timeout"))

    payload = {"query": "crash test", "top_k": 5}
    response = client.post("/search", json=payload)

    assert response.status_code == 500
    assert "Connection Timeout" in response.json()["message"]