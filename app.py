from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import config
from src.prompt import system_prompt

def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)

    if not config.is_valid:
        raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in environment.")

    embeddings = download_hugging_face_embeddings()
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.pinecone_index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    chat_model = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    @app.route("/")
    def index() -> str:
        """Render the main chat interface."""
        return render_template('chat.html')

    @app.route("/get", methods=["GET", "POST"])
    def chat() -> str:
        """Handle incoming chat messages and return RAG-based responses."""
        msg: str = request.form.get("msg", "")
        if not msg:
            return "Please provide a valid message."
            
        print(f"User Input: {msg}")
        response = rag_chain.invoke({"input": msg})
        
        answer: str = str(response.get("answer", "No answer found."))
        print(f"Response: {answer}")
        
        return answer

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)

