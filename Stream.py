import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import io
import uuid
from pathlib import Path

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)

# Page configuration
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("Chat with PDF using Hybrid Search")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.collection_name = None
        st.rerun()

def process_pdf(file_bytes):
    """Extract and chunk text from PDF."""
    try:
        # Extract text
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Split text using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        return chunks
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

def initialize_rag_system(chunks):
    """Initialize the RAG system with chunked text."""
    try:
        # Initialize client
        client = QdrantClient(":memory:")
        
        # Initialize embedding models
        dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        sparse_model = SparseTextEmbedding("Qdrant/bm25")
        colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        
        # Generate embeddings for chunks
        dense_embeddings = list(dense_model.embed(chunks))
        sparse_embeddings = list(sparse_model.embed(chunks))
        colbert_embeddings = list(colbert_model.embed(chunks))
        
        # Create collection with hybrid configuration
        collection_name = f"pdf-docs-{str(uuid.uuid4())}"
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "miniLM": VectorParams(
                    size=384,
                    distance=Distance.COSINE
                ),
                "colbert": VectorParams(
                    size=128,
                    distance=Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams()
            }
        )
        
        # Prepare points for insertion
        points = [
            PointStruct(
                id=idx,
                vector={
                    "miniLM": dense_embedding,
                    "bm25": sparse_embedding.as_object(),
                    "colbert": colbert_embedding
                },
                payload={"text": chunk, "chunk_id": idx}
            )
            for idx, (dense_embedding, sparse_embedding, colbert_embedding, chunk) 
            in enumerate(zip(dense_embeddings, sparse_embeddings, colbert_embeddings, chunks))
        ]
        
        # Insert data
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return client, dense_model, sparse_model, colbert_model, collection_name
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        raise e

def perform_hybrid_search(query, client, dense_model, sparse_model, colbert_model, collection_name):
    """Perform hybrid search on the chunked text."""
    # Generate query embeddings
    dense_query = next(dense_model.query_embed(query))
    sparse_query = next(sparse_model.query_embed(query))
    colbert_query = next(colbert_model.query_embed(query))
    
    # Hybrid search with reranking
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using="miniLM",
                limit=5
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query.as_object()),
                using="bm25",
                limit=5
            )
        ],
        query=colbert_query,
        using="colbert",
        limit=3,
        with_payload=True
    )
    
    return results

# Process uploaded PDF
if uploaded_file is not None:
    # Check if we need to process the file
    if st.session_state.collection_name is None:
        with st.spinner("Processing PDF..."):
            try:
                # Get file content and process into chunks
                file_bytes = uploaded_file.getvalue()
                chunks = process_pdf(file_bytes)
                
                # Initialize RAG system with chunks
                client, dense_model, sparse_model, colbert_model, collection_name = initialize_rag_system(chunks)
                
                # Store in session state
                st.session_state.client = client
                st.session_state.dense_model = dense_model
                st.session_state.sparse_model = sparse_model
                st.session_state.colbert_model = colbert_model
                st.session_state.collection_name = collection_name
                
                st.success(f"✅ Processed {len(chunks)} chunks successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "excerpts" in message:
            with st.expander("View Sources"):
                for text, score in message["excerpts"]:
                    st.write(f"Score: {score:.4f}")
                    st.write(f"Text: {text}")
                    st.write("-" * 50)

# Chat input
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.collection_name is None:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF file first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF file first."})
    else:
        with st.spinner("Thinking..."):
            try:
                # Get relevant chunks
                results = perform_hybrid_search(
                    prompt,
                    st.session_state.client,
                    st.session_state.dense_model,
                    st.session_state.sparse_model,
                    st.session_state.colbert_model,
                    st.session_state.collection_name
                )
                
                # Extract results
                points_list = results.model_dump()['points']
                extracted_data = [(point['payload']['text'], point['score']) for point in points_list]
                
                # Generate context
                context = "\n".join([text for text, _ in extracted_data])
                
                # Prepare prompt for LLM
                full_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"""
                
                # Get response from LLM
                response = abc_response(full_prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                    with st.expander("View Sources"):
                        for text, score in extracted_data:
                            st.write(f"Score: {score:.4f}")
                            st.write(f"Text: {text}")
                            st.write("-" * 50)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "excerpts": extracted_data
                })
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
