# Medical Chatbot Architecture

## Overview
The Medical Chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware medical answers based on a curated knowledge base (PDFs) and real-time web search.

## Core Components
1. **Flask Application**: Serves as the backend web framework and API provider (`app.py`).
2. **LangChain Agent**: Uses the `create_tool_calling_agent` pattern powered by OpenAI's `gpt-4o`.
3. **Pinecone Vector Store**: Stores text embeddings of medical PDFs for semantic similarity search.
4. **Tavily Search**: Integrated as a fallback web-search tool to retrieve real-time medical facts outside the provided PDFs.

## How it Works
- The user inputs a message via the web UI.
- The Flask app receives the message at the `/get` endpoint.
- The LangChain Agent (`AgentExecutor`) processes the query.
- The agent first attempts to use the `local_medical_database` (Pinecone) tool, applying similarity search (fetching the top 3 results `k=3`).
- If more information is needed, the `TavilySearchResults` tool is invoked (fetching up to 2 web results).
- The `gpt-4o` model synthesizes a final response and returns it to the frontend.
