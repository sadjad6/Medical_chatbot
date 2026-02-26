import pytest
from unittest.mock import patch, MagicMock
from app import create_app
from config import config

@pytest.fixture
def mock_langchain_pinecone():
    with patch('app.download_hugging_face_embeddings') as mock_hugging_face, \
         patch('app.PineconeVectorStore') as mock_pinecone, \
         patch('app.ChatOpenAI') as mock_openai, \
         patch('app.create_retrieval_chain') as mock_retrieval, \
         patch('app.create_stuff_documents_chain') as mock_stuff:
         
        # Mock RAG chain invoke
        mock_rag_chain = MagicMock()
        mock_retrieval.return_value = mock_rag_chain
        mock_rag_chain.invoke.return_value = {"answer": "Mocked test answer."}
        
        yield mock_rag_chain

@pytest.fixture
def client(monkeypatch, mock_langchain_pinecone):
    """Setup Flask test client with mock environment."""
    monkeypatch.setattr(config, "is_valid", True)
    
    app = create_app()
    app.config.update({"TESTING": True})
    
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test the main HTML page is served."""
    response = client.get("/")
    assert response.status_code == 200

def test_chat_empty_message(client):
    """Unhappy path test: Sending an empty message."""
    response = client.post("/get", data={"msg": ""})
    assert response.status_code == 200
    assert b"Please provide a valid message." in response.data

def test_chat_missing_msg_field(client):
    """Unhappy path test: Missing 'msg' in form data entirely."""
    response = client.post("/get", data={})
    assert response.status_code == 200
    assert b"Please provide a valid message." in response.data

def test_missing_environment_variables(monkeypatch):
    """Unhappy path test: Ensure ValueError is raised when config is missing."""
    monkeypatch.setattr(config, "is_valid", False)
    
    with pytest.raises(ValueError, match="Missing PINECONE_API_KEY or OPENAI_API_KEY"):
        create_app()

def test_valid_chat_message(client, mock_langchain_pinecone):
    """Happy path test: Valid query returns proper answer."""
    response = client.post("/get", data={"msg": "What is acne?"})
    assert response.status_code == 200
    assert b"Mocked test answer." in response.data
    mock_langchain_pinecone.invoke.assert_called_once_with({"input": "What is acne?"})
