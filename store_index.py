from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from config import config

def main() -> None:
    """Store raw PDF data into Pinecone vector index."""
    if not config.is_valid:
        raise ValueError("Missing required API keys in environment.")

    extracted_data = load_pdf_file('data/')
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)
    
    embeddings = download_hugging_face_embeddings()
    
    pc = Pinecone(api_key=config.pinecone_api_key)
    
    if not pc.has_index(config.pinecone_index_name):
        pc.create_index(
            name=config.pinecone_index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
    # Store documents
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=config.pinecone_index_name,
        embedding=embeddings, 
    )
    print(f"Successfully processed and stored {len(text_chunks)} chunks into Pinecone index '{config.pinecone_index_name}'.")

if __name__ == '__main__':
    main()