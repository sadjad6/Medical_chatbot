# Medical Chatbot

A comprehensive Medical Chatbot built with Python, LangChain, Pinecone, OpenAI, and Flask. This project leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware medical answers based on a provided knowledge base.

## Quick Start (with `uv`)

This project is built and optimized for **Python 3.12** using **uv** for dependency management.

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sadjad6/Medical_chatbot.git
   cd Medical_chatbot
   ```

2. **Create and activate the environment:**
   ```bash
   uv venv --python 3.12
   # On Windows: .venv\Scripts\activate
   # On macOS/Linux: source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Pinecone & OpenAI credentials:
   ```ini
   PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

5. **Store Embeddings to Pinecone:**
   ```bash
   uv run python store_index.py
   ```

6. **Run the Application locally:**
   ```bash
   uv run python app.py
   ```
   Open your browser at `http://localhost:8080` to access the chat UI.
