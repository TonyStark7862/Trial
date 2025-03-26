import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import pickle
import hashlib
import os
from pathlib import Path
import io
import base64
from datetime import datetime

# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant instead

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Custom CSS
def load_css():
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        .stChat {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            padding: 1rem;
            margin: 1rem 0;
            background-color: white;
        }
        .upload-section {
            border: 2px dashed #aaa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        .metrics-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize app
st.set_page_config(
    page_title="Advanced RAG PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# Page header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>📚 Advanced PDF Chat Assistant</h1>
        <p style='color: #666;'>Upload multiple PDFs and chat with your documents using advanced RAG</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # PDF Management Section
    st.markdown("#### 📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files"
    )
    
    if uploaded_files:
        st.success(f"📥 {len(uploaded_files)} files uploaded")
        
        # Display uploaded files with timestamps
        st.markdown("#### 📋 Uploaded Documents")
        for file in uploaded_files:
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.write(datetime.now().strftime("%H:%M"))
    
    # Chat Management
    st.markdown("#### 💬 Chat Management")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if USE_FAISS:
            st.session_state.retriever = None
        else:
            st.session_state.collection_name = None
        st.success("Chat history cleared!")
        st.rerun()
    
    # Display Metrics
    st.markdown("#### 📊 Chat Metrics")
    if "messages" in st.session_state:
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
            <div class='metrics-card'>
                <p>Total Messages: {total_messages}</p>
                <p>User Queries: {user_messages}</p>
            </div>
        """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if USE_FAISS and "retriever" not in st.session_state:
    st.session_state.retriever = None
if not USE_FAISS and "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Keep the existing helper functions (load_sentence_transformer, setup_qdrant_client, etc.)
[Previous helper functions remain exactly the same]

# Modified PDF processing to handle multiple files
def process_multiple_pdfs(uploaded_files):
    combined_hash = ""
    combined_text = ""
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        combined_hash += file_hash
        
        # Extract text from PDF
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            combined_text += page.extract_text() + "\n"
            
        st.sidebar.success(f"✅ Processed: {uploaded_file.name}")
    
    final_hash = hashlib.md5(combined_hash.encode()).hexdigest()
    return final_hash, combined_text

# Process uploaded PDFs
if uploaded_files:
    combined_hash, combined_text = process_multiple_pdfs(uploaded_files)
    
    if USE_FAISS:
        # FAISS Implementation for multiple PDFs
        cache_file = VECTORDB_DIR / f"{combined_hash}.pkl"
        
        if st.session_state.retriever is None:
            with st.spinner("Processing PDFs with FAISS..."):
                # Load from cache if exists
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            vector_store = pickle.load(f)
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.success("✅ Loaded from cache")
                    except Exception as e:
                        st.error(f"Error loading cache: {e}")
                
                # Process if not in cache
                if st.session_state.retriever is None:
                    try:
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(combined_text)
                        
                        # Create embeddings and vector store
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-mpnet-base-v2"
                        )
                        
                        vector_store = FAISS.from_texts(chunks, embeddings)
                        
                        # Save to cache
                        with open(cache_file, "wb") as f:
                            pickle.dump(vector_store, f)
                        
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.success("✅ PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
    else:
        # Qdrant Implementation for multiple PDFs
        collection_name = f"pdf_{combined_hash}"
        
        if st.session_state.collection_name != collection_name:
            with st.spinner("Processing PDFs with Qdrant..."):
                qdrant_client = setup_qdrant_client()
                model = load_sentence_transformer()
                
                if qdrant_client:
                    # Create collection
                    vector_size = model.get_sentence_embedding_dimension()
                    create_collection(qdrant_client, collection_name, vector_size)
                    
                    # Process text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(combined_text)
                    
                    # Create embeddings
                    embeddings = model.encode(chunks)
                    
                    # Prepare points
                    points = [
                        models.PointStruct(
                            id=idx,
                            vector=embedding.tolist(),
                            payload={"text": chunk, "chunk_id": idx}
                        )
                        for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
                    ]
                    
                    # Upload to collection
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    
                    st.session_state.collection_name = collection_name
                    st.success("✅ PDFs processed successfully!")

# Enhanced chat interface
st.markdown("### 💬 Chat Interface")

# Display chat messages with enhanced styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            <strong>Source {i+1}:</strong><br>{source}
                        </div>
                    """, unsafe_allow_html=True)

# Enhanced chat input
if prompt := st.chat_input("💭 Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if documents are uploaded
    if (USE_FAISS and st.session_state.retriever is None) or \
       (not USE_FAISS and st.session_state.collection_name is None):
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload PDF documents first.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Please upload PDF documents first."
        })
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get relevant chunks
                if USE_FAISS:
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    sources = [doc.page_content for doc in docs]
                else:
                    sources = search_chunks(st.session_state.collection_name, prompt)
                
                if not sources:
                    response = "I couldn't find relevant information in the documents. Please try rephrasing your question."
                else:
                    context = "\n\n".join(sources)
                    full_prompt = f"""
                    Answer the following question based on the provided context.
                    
                    Context:
                    {context}
                    
                    Question: {prompt}
                    
                    Answer:
                    """
                    
                    # Get response from custom LLM function
                    response = abc_response(full_prompt) if 'abc_response' in globals() else \
                             f"Using local LLM to answer: {prompt}\n\nBased on the documents, I found relevant information that would help answer this question."
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response)
                    if sources:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(sources):
                                st.markdown(f"""
                                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                        <strong>Source {i+1}:</strong><br>{source}
                                    </div>
                                """, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources if sources else []
                })
                
            except Exception as e:
                error_message = f"❌ Error generating response: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
