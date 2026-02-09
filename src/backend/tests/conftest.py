import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
# We import app inside the fixture to ensure patches apply before import

@pytest.fixture
def mock_pinecone_db():
    """
    Mock the index operations (query, stats).
    """
    mock_index = AsyncMock()
    # Mock search results
    mock_index.query.return_value = {"matches": []}
    # Mock stats
    mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=1234)
    return mock_index

@pytest.fixture
def mock_encoder():
    """
    Manually inject a mock into the Singleton _instance.
    This bypasses __init__ entirely, so no ONNX files are needed.
    """
    mock_instance = MagicMock()
    mock_instance.encode_text.return_value = [0.0] * 512
    mock_instance.warm_up.return_value = None
    
    # Import inside fixture to avoid early init
    from encoder import CLIPTextEncoder
    
    # Save original
    previous_instance = CLIPTextEncoder._instance
    # Inject mock
    CLIPTextEncoder._instance = mock_instance
    
    yield mock_instance
    
    # Restore original
    CLIPTextEncoder._instance = previous_instance

@pytest.fixture(autouse=True)
def mock_pinecone_client_init():
    """
    Automatically patch the 'pinecone.Pinecone' class GLOBALLY.
    This works regardless of whether main.py uses 'import pinecone' 
    or 'from pinecone import Pinecone'.
    """
    with patch("pinecone.Pinecone") as MockClient:
        # Ensure that when pc.Index("name") is called, it returns our AsyncMock
        MockClient.return_value.Index.return_value = AsyncMock()
        yield MockClient

@pytest.fixture
def client(mock_pinecone_db, mock_encoder, mock_pinecone_client_init):
    """
    Returns a TestClient with all dependencies mocked.
    """
    from main import app, get_db
    
    # Override the get_db dependency to return our specific mock index
    app.dependency_overrides[get_db] = lambda: mock_pinecone_db
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()