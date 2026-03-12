import pytest
from unittest.mock import patch, MagicMock
from app import create_app
from config import config

@pytest.fixture
def mock_agent_executor():
    with patch('app.download_hugging_face_embeddings') as mock_hugging_face, \
         patch('app.PineconeVectorStore') as mock_pinecone, \
         patch('app.ChatOpenAI') as mock_openai, \
         patch('app.create_tool_calling_agent') as mock_agent, \
         patch('app.AgentExecutor') as mock_executor_cls, \
         patch('app.TavilySearchResults') as mock_tavily:
         
        # Mock AgentExecutor invoke
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_executor_instance.invoke.return_value = {"output": "Mocked agent answer."}
        
        yield mock_executor_instance

@pytest.fixture
def client(monkeypatch, mock_agent_executor):
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

def test_valid_chat_message(client, mock_agent_executor):
    """Happy path test: Valid query returns proper answer."""
    response = client.post("/get", data={"msg": "What is acne?"})
    assert response.status_code == 200
    assert b"Mocked agent answer." in response.data
    mock_agent_executor.invoke.assert_called_once_with({"input": "What is acne?"})
