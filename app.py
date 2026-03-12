from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from config import config
from src.prompt import agent_prompt

def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)

    if not config.is_valid:
        raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in environment.")
        
    if not config.tavily_api_key:
        print("WARNING: TAVILY_API_KEY missing. Web Search Tool will fail if invoked.")

    embeddings = download_hugging_face_embeddings()
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.pinecone_index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    pinecone_tool = create_retriever_tool(
        retriever,
        "local_medical_database",
        "Search for curated medical knowledge from the provided PDFs. Always try this tool first."
    )
    
    web_tool = TavilySearchResults(max_results=2)
    
    tools = [pinecone_tool, web_tool]
    
    chat_model = ChatOpenAI(model="gpt-4o")
    agent = create_tool_calling_agent(chat_model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    @app.route("/")
    def index() -> str:
        """Render the main chat interface."""
        return render_template('chat.html')

    @app.route("/get", methods=["GET", "POST"])
    def chat() -> str:
        """Handle incoming chat messages and return Agent-driven responses."""
        msg: str = request.form.get("msg", "")
        if not msg:
            return "Please provide a valid message."
            
        print(f"User Input: {msg}")
        response = agent_executor.invoke({"input": msg})
        
        answer: str = str(response.get("output", "No answer found."))
        print(f"Response: {answer}")
        
        return answer

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)

