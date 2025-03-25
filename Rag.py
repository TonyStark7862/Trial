"""
Hybrid RAG Implementation
-------------------------
This script demonstrates a complete implementation of a hybrid RAG system
using multiple embedding models for optimal retrieval performance.
Processes PDF documents from a local path and indexes them with metadata
including page numbers for source tracking.
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path

# PDF processing
import fitz  # PyMuPDF

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import qdrant_client.models as models
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding

# Initialize client with local persistence
client = QdrantClient(path="./qdrant_storage")  # Local persistence

def setup_collection(collection_name: str = "hybrid-search"):
    """
    Create a collection with the necessary vector configurations
    """
    # Check if collection exists and delete if it does
    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        client.delete_collection(collection_name)
    
    # Create a new collection with vector configurations for all embedding types
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=384,  # Dimension of all-MiniLM-L6-v2
                distance=models.Distance.COSINE,
            ),
            "bm25": models.VectorParams(
                size=32_768,  # Common sparse vector dimension
                distance=models.Distance.DOT,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,  # Dimension of colbertv2.0
                distance=models.Distance.COSINE,
            ),
        },
    )
    print(f"Collection '{collection_name}' created successfully")

def load_embedding_models():
    """
    Load all required embedding models
    """
    # Load dense embedding model for semantic understanding
    dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load sparse embedding model for keyword matching
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    
    # Load late interaction embedding model for contextual precision
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    
    return dense_embedding_model, bm25_embedding_model, late_interaction_embedding_model

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF along with page numbers and title only
    """
    document = fitz.open(pdf_path)
    chunks = []
    
    # Get only the title from metadata
    title = document.metadata.get("title", "Unknown")
    if title == "Unknown" or not title.strip():
        # Use filename as title if no title metadata
        title = Path(pdf_path).stem
    
    for page_num, page in enumerate(document):
        text = page.get_text()
        # Skip nearly empty pages
        if len(text) > 10:  # Only include non-empty pages
            chunks.append({
                "content": text,
                "page_num": page_num + 1,  # 1-based page numbering
                "title": title
            })
    
    return chunks

def process_pdf_directory(pdf_dir, 
                        dense_model, 
                        bm25_model, 
                        late_model, 
                        collection_name: str = "hybrid-search"):
    """
    Process all PDFs in a directory and index them with metadata
    """
    all_chunks = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        chunks = extract_text_from_pdf(str(pdf_path))
        all_chunks.extend(chunks)
    
    print(f"Extracted {len(all_chunks)} text chunks from PDFs")
    
    # Generate document texts for embedding
    documents = [chunk["content"] for chunk in all_chunks]
    
    # Generate embeddings for all documents
    print("Generating dense embeddings...")
    dense_embeddings = list(dense_model.embed(documents))
    
    print("Generating BM25 sparse embeddings...")
    bm25_embeddings = list(bm25_model.embed(documents))
    
    print("Generating late interaction embeddings...")
    late_interaction_embeddings = list(late_model.embed(documents))
    
    # Create points with all embeddings and simplified metadata
    print("Creating points with all embeddings and simplified metadata...")
    points = []
    for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, chunk) in enumerate(
        zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, all_chunks)
    ):
        point = PointStruct(
            id=idx,
            vector={
                "all-MiniLM-L6-v2": dense_embedding,
                "bm25": bm25_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding,
            },
            payload={
                "content": chunk["content"],
                "page_num": chunk["page_num"],
                "title": chunk["title"],
                "filename": Path(pdf_path).name  # Keep just the filename
            }
        )
        points.append(point)
    
    # Upload points to the collection
    print(f"Uploading {len(points)} points to collection '{collection_name}'...")
    operation_info = client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Documents processed and stored. Operation ID: {operation_info.operation_id}")
    return operation_info

def hybrid_search(query: str, 
                 dense_model, 
                 bm25_model, 
                 late_model, 
                 collection_name: str = "hybrid-search", 
                 limit: int = 10):
    """
    Perform hybrid search using all three embedding types
    """
    print(f"Searching for: '{query}'")
    
    # Generate query embeddings
    dense_vectors = next(dense_model.query_embed(query))
    sparse_vectors = next(bm25_model.query_embed(query))
    late_vectors = next(late_model.query_embed(query))
    
    # Set up prefetch parameter for hybrid search
    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]
    
    # Execute search with re-ranking
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=limit,
    )
    
    return results

def format_context_from_results(results):
    """
    Format search results into context for LLM with simplified source information
    """
    context = ""
    sources = []
    
    for i, result in enumerate(results):
        content = result.payload.get("content", "")
        page_num = result.payload.get("page_num", "?")
        title = result.payload.get("title", "Unknown")
        filename = result.payload.get("filename", "Unknown")
        score = result.score
        
        # Add formatted content
        context += f"\n[CONTENT {i+1}] {content}\n"
        
        # Track source with simplified information
        source = {
            "title": title,
            "filename": filename,
            "page_num": page_num,
            "score": round(score, 4),
            "content_snippet": content[:100] + "..." if len(content) > 100 else content
        }
        sources.append(source)
    
    return context, sources

def generate_response(query, context, sources):
    """
    Generate response using external LLM function
    """
    # Construct prompt with context
    prompt = f"""
Please answer the following question based only on the provided context:

QUESTION: {query}

CONTEXT:
{context}

Provide a comprehensive answer using only the information in the context.
"""
    
    # Using the external abc_response function as requested
    # Note: This function should be defined elsewhere
    llm_response = abc_response(prompt)
    
    return llm_response, sources

def abc_response(prompt):
    """
    Placeholder for external LLM function
    This would be replaced by the actual function provided elsewhere
    """
    # Normally this would call the external LLM API
    # For this implementation, we're just leaving it as a placeholder
    return "This is a placeholder for the LLM response. Replace with actual abc_response function."

def query_system(query, pdf_dir=None, dense_model=None, bm25_model=None, late_model=None):
    """
    End-to-end query system for hybrid RAG
    """
    # Load models if not provided
    if not dense_model or not bm25_model or not late_model:
        dense_model, bm25_model, late_model = load_embedding_models()
    
    # Perform hybrid search
    results = hybrid_search(query, dense_model, bm25_model, late_model)
    
    # Format context and get sources
    context, sources = format_context_from_results(results)
    
    # Generate LLM response
    llm_response, sources = generate_response(query, context, sources)
    
    # Display results with simplified sources
    print("\n===== LLM RESPONSE =====")
    print(llm_response)
    print("\n===== SOURCES =====")
    for i, source in enumerate(sources):
        print(f"{i+1}. Title: {source['title']} | Page: {source['page_num']} | Relevance: {source['score']}")
        print(f"   File: {source['filename']}")
        print(f"   Snippet: {source['content_snippet']}\n")
    
    return {
        "query": query,
        "response": llm_response,
        "sources": sources
    }

def main():
    """
    Main function demonstrating the complete hybrid RAG workflow
    """
    # Set path to PDF directory
    pdf_dir = "./documents"  # Change this to your PDF directory path
    
    # Set up collection
    setup_collection()
    
    # Load embedding models
    dense_model, bm25_model, late_model = load_embedding_models()
    
    # Check if we should index documents
    should_index = input("Index documents from PDF directory? (y/n): ").lower() == 'y'
    
    if should_index:
        # Process and store PDFs
        process_pdf_directory(pdf_dir, dense_model, bm25_model, late_model)
    
    # Interactive query loop
    while True:
        # Get query from user
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        # Process query
        query_system(query, pdf_dir, dense_model, bm25_model, late_model)

if __name__ == "__main__":
    main()
