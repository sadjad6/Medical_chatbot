import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Store environment variables and application configurations."""
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_index_name: str = "medical-chatbot"
    
    @property
    def is_valid(self) -> bool:
        """Check if all required API keys are present."""
        return bool(self.pinecone_api_key and self.openai_api_key)

config = Config()

# Update os environment so existing langchain setups implicitly pick it up
os.environ["PINECONE_API_KEY"] = config.pinecone_api_key
os.environ["OPENAI_API_KEY"] = config.openai_api_key
