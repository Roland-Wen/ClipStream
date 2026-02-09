import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app, get_db
from encoder import CLIPTextEncoder  # Import the real class

@pytest.fixture
def mock_pinecone_db():
    """Mock Pinecone database calls."""
    mock_index = AsyncMock()
    mock_index.query.return_value = {"matches": []}
    mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=1234)
    return mock_index

@pytest.fixture
def mock_encoder():
    """
    Manually inject a mock into the Singleton _instance.
    This bypasses __init__ entirely, so no files are needed.
    """
    # 1. Create a Fake Encoder
    mock_instance = MagicMock()
    mock_instance.encode_text.return_value = [0.0] * 512
    mock_instance.warm_up.return_value = None
    
    # 2. Inject it directly into the class
    # Because 'main.CLIPTextEncoder' points to this same class, 
    # main.py will now see this mock.
    previous_instance = CLIPTextEncoder._instance
    CLIPTextEncoder._instance = mock_instance
    
    yield mock_instance
    
    # 3. Cleanup: Reset so we don't break other tests
    CLIPTextEncoder._instance = previous_instance

@pytest.fixture
def client(mock_pinecone_db, mock_encoder):
    """
    Returns a TestClient with dependencies mocked.
    """
    app.dependency_overrides[get_db] = lambda: mock_pinecone_db
    
    # We don't even need to override get_encoder here because
    # we hacked the Singleton itself!
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()