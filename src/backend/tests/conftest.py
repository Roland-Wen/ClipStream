import sys
from unittest.mock import MagicMock, AsyncMock

# --- 1. THE NUCLEAR OPTION: Mock 'pinecone' before ANY import happens ---
mock_pinecone_module = MagicMock()

# Mock the class factory
mock_pinecone_class = MagicMock()

# IMPORTANT FIX 1: Map BOTH 'Pinecone' and 'PineconeAsyncio' to the same mock class
# Your database.py imports PineconeAsyncio, so we must ensure that name exists.
mock_pinecone_module.Pinecone = mock_pinecone_class
mock_pinecone_module.PineconeAsyncio = mock_pinecone_class

# Create the specific index mock that acts as the "Async Engine"
mock_index = AsyncMock()

# Create the data object for stats
stats_response = MagicMock(total_vector_count=1234)

# Make BOTH query and describe_index_stats AsyncMocks so they can be awaited
mock_index.query = AsyncMock(return_value={"matches": []})
mock_index.describe_index_stats = AsyncMock(return_value=stats_response)

# IMPORTANT: Set up the client instance (what PineconeAsyncio() returns)
# It needs describe_index(), IndexAsyncio(), and close() to be async
mock_client_instance = MagicMock()

# Mock describe_index to return an object with .host attribute
description_response = MagicMock(host="test-host.pinecone.io")
mock_client_instance.describe_index = AsyncMock(return_value=description_response)

# Mock IndexAsyncio to return our mock_index
mock_client_instance.IndexAsyncio = MagicMock(return_value=mock_index)

# Mock close as AsyncMock
mock_client_instance.close = AsyncMock()

# Make PineconeAsyncio() return our configured mock client
mock_pinecone_class.return_value = mock_client_instance

# IMPORTANT FIX 2: Mock 'IndexAsyncio'
# Your database.py calls `client.IndexAsyncio(...)`, not `client.Index(...)`
# We map both just to be safe.
mock_pinecone_class.return_value.Index.return_value = mock_index
mock_pinecone_class.return_value.IndexAsyncio.return_value = mock_index

# INJECT INTO SYSTEM
sys.modules["pinecone"] = mock_pinecone_module


# --- 2. NOW we can import the rest safely ---
import pytest
import os
from fastapi.testclient import TestClient

# Set dummy env vars so Pydantic settings don't crash
os.environ["PINECONE_API_KEY"] = "test_key_only"
os.environ["PINECONE_INDEX_NAME"] = "test_index"


# --- 3. FIXTURES ---

@pytest.fixture
def mock_pinecone_db():
    """
    Returns the same mock index we injected globally, 
    so tests can assert against it.
    """
    return mock_index

@pytest.fixture
def mock_encoder():
    """
    Mock the Encoder to bypass ONNX file loading.
    Patches it in main.py where it's used.
    """
    from unittest.mock import patch

    mock_instance = MagicMock()
    # Mock encode_text to return a standard list of floats
    mock_instance.encode_text.return_value = [0.0] * 512
    mock_instance.warm_up.return_value = None

    # Patch where it's imported in main.py
    with patch("main.CLIPTextEncoder") as MockClass:
        MockClass.get_instance.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def client(mock_pinecone_db, mock_encoder):
    """
    Test Client.
    """
    # Import app INSIDE the fixture (Lazy Import)
    from main import app, get_db

    # Override dependency: get_db is async, so the override must be too
    async def mock_get_db():
        return mock_pinecone_db

    app.dependency_overrides[get_db] = mock_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()