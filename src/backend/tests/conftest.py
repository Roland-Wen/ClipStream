import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app, get_db, get_encoder

# 1. Mock the Pinecone Database
@pytest.fixture
def mock_pinecone_db():
    """
    Creates a fake Pinecone index that returns canned responses.
    """
    mock_index = AsyncMock()
    
    # Default behavior: Return empty matches
    mock_index.query.return_value = {"matches": []}
    
    return mock_index

# 2. Mock the ONNX Encoder
@pytest.fixture
def mock_encoder():
    """
    Creates a fake Encoder that returns a fixed 512-dim vector
    without actually running ONNX Runtime.
    """
    encoder = MagicMock()
    # Return a dummy vector of 512 zeros
    encoder.encode_text.return_value = [0.0] * 512
    # Mock warmup so it doesn't crash
    encoder.warm_up = MagicMock()
    return encoder

# 3. Create the Test Client with Overrides
@pytest.fixture
def client(mock_pinecone_db, mock_encoder):
    """
    Returns a FastAPI TestClient with dependencies swapped out for mocks.
    """
    # Override the dependency injection
    app.dependency_overrides[get_db] = lambda: mock_pinecone_db
    app.dependency_overrides[get_encoder] = lambda: mock_encoder
    
    # Create client
    with TestClient(app) as c:
        yield c
    
    # Cleanup overrides after test
    app.dependency_overrides.clear()