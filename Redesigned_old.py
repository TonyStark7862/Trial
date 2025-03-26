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
import json
import time
import uuid
from datetime import datetime

# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant instead

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# User collections directory
COLLECTIONS_DIR = Path("./user_collections")
COLLECTIONS_DIR.mkdir(exist_ok=True, parents=True)

# App configuration and styling
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stRadio > div {
        padding: 10px;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
    }
    .stRadio label {
        font-weight: 500;
        color: #1F2937;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #34D399;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton button {
        background-color: #1E40AF;
        color: white;
        font-weight: 500;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #1E3A8A;
    }
    .delete-btn button {
        background-color: #DC2626;
    }
    .delete-btn button:hover {
        background-color: #B91C1C;
    }
    .clear-btn button {
        background-color: #4B5563;
    }
    .clear-btn button:hover {
        background-color: #374151;
    }
    .file-uploader {
        border: 2px dashed #CBD5E1;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
        font-weight: 500;
        color: #1E3A8A;
    }
    .chat-container {
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #F9FAFB;
    }
    .collection-info {
        font-size: 0.9rem;
        color: #6B7280;
        font-style: italic;
    }
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .sources-container {
        background-color: #F3F4F6;
        border-radius: 0.375rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-header">📚 DocuMind AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Personal Document Knowledge Base with AI-powered Search</p>', unsafe_allow_html=True)

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "upload"  # Default mode

# Get user collections file path
def get_user_collections_path():
    return COLLECTIONS_DIR / f"user_{st.session_state.user_id}.json"

# Save user collections
def save_user_collections(collections):
    collections_path = get_user_collections_path()
    with open(collections_path, "w") as f:
        json.dump(collections, f)

# Load user collections
def load_user_collections():
    collections_path = get_user_collections_path()
    if collections_path.exists():
        with open(collections_path, "r") as f:
            return json.load(f)
    return {}

# Initialize the embedding model for Qdrant
@st.cache_resource
def load_sentence_transformer():
    """Initialize the embedding model from local path or download if not present."""
    try:
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        return model
    except Exception as e:
        with st.spinner("Downloading embedding model (this may take a moment)..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # Save the model for future use
            os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
            model.save(LOCAL_MODEL_PATH)
            return model

# Setup Qdrant client
@st.cache_resource
def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    try:
        client = QdrantClient(path=str(VECTORDB_DIR / "qdrant_db"))
        return client
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

# Create collection for the PDF if it doesn't exist (Qdrant)
def create_collection(client, collection_name, vector_size=384):
    """Create a new collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return True
        return False
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        return False

# Process PDF and add to Qdrant
def process_pdf_qdrant(file_bytes, collection_name, filename):
    try:
        # Extract text
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Create embeddings for chunks
        embeddings = model.encode(chunks)
        
        # Prepare points for upload with file source information
        points = [
            models.PointStruct(
                id=idx + 10000,  # Adding offset to avoid ID conflicts
                vector=embedding.tolist(),
                payload={
                    "text": chunk, 
                    "chunk_id": idx,
                    "source": filename,
                    "timestamp": int(time.time())
                }
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        
        # Upload to collection
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return len(points)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Search for relevant chunks in Qdrant
def search_chunks(collection_name, query, limit=4):
    """Search for chunks similar to the query."""
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query])[0]
        
        # Search in collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            with_payload=True
        )
        
        # Return text and source information
        return [{"text": result.payload["text"], "source": result.payload.get("source", "Unknown")} for result in search_results]
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

# Delete a collection
def delete_collection(collection_name):
    try:
        qdrant_client = setup_qdrant_client()
        qdrant_client.delete_collection(collection_name=collection_name)
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=DocuMind", width=150)
    st.markdown("### Navigation")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["📄 Upload Documents", "💬 Chat with Documents"]
    )
    
    # Update session state based on mode selection
    if mode == "📄 Upload Documents" and st.session_state.app_mode != "upload":
        st.session_state.app_mode = "upload"
        st.rerun()
    elif mode == "💬 Chat with Documents" and st.session_state.app_mode != "chat":
        st.session_state.app_mode = "chat"
        st.rerun()
    
    st.markdown("---")
    
    # Display user ID in a clean format
    st.markdown(f"**Session ID:** {st.session_state.user_id[:8]}...")
    
    # Clear session button
    if st.button("🔄 New Session", help="Start a new session with a fresh user ID"):
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.current_collection = None
        st.rerun()

# Upload Mode
if st.session_state.app_mode == "upload":
    st.markdown('<h2 class="section-header">📄 Upload Documents to Create Knowledge Base</h2>', unsafe_allow_html=True)
    
    # Collection name input
    collection_name = st.text_input(
        "Knowledge Base Name",
        help="Give a unique name to your collection of documents",
        placeholder="E.g., Project Research, Company Docs, Study Materials"
    )
    
    # PDF uploader with formatting
    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    st.markdown("### Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Select multiple PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_files and collection_name:
        # Process button
        process_col1, process_col2 = st.columns([1, 6])
        with process_col1:
            process_button = st.button("🚀 Process Documents", use_container_width=True)
        
        if process_button:
            if not collection_name.strip():
                st.markdown('<div class="error-box">Please provide a name for your knowledge base.</div>', unsafe_allow_html=True)
            else:
                # Sanitize collection name for Qdrant
                sanitized_name = f"user_{st.session_state.user_id}_{collection_name.lower().replace(' ', '_')}"
                
                # Load user collections
                user_collections = load_user_collections()
                
                # Check if collection already exists
                if sanitized_name in user_collections:
                    st.markdown('<div class="error-box">A knowledge base with this name already exists. Please choose a different name.</div>', unsafe_allow_html=True)
                else:
                    # Initialize the model and client
                    with st.spinner("Initializing..."):
                        model = load_sentence_transformer()
                        qdrant_client = setup_qdrant_client()
                        vector_size = model.get_sentence_embedding_dimension()
                        
                        # Create collection
                        is_new = create_collection(qdrant_client, sanitized_name, vector_size)
                        
                        # Process each PDF file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_chunks = 0
                        for i, uploaded_file in enumerate(uploaded_files):
                            file_bytes = uploaded_file.getvalue()
                            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                            try:
                                chunks_added = process_pdf_qdrant(file_bytes, sanitized_name, uploaded_file.name)
                                total_chunks += chunks_added
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {e}")
                                continue
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Save collection metadata
                        user_collections[sanitized_name] = {
                            "display_name": collection_name,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "files": [f.name for f in uploaded_files],
                            "chunks": total_chunks
                        }
                        save_user_collections(user_collections)
                        
                        # Success message
                        progress_bar.empty()
                        status_text.empty()
                        st.markdown(f'<div class="success-box">✅ Successfully created knowledge base "{collection_name}" with {len(uploaded_files)} documents and {total_chunks} text chunks.</div>', unsafe_allow_html=True)
                        
                        # Add collection switch button
                        if st.button("💬 Start Chatting with this Knowledge Base"):
                            st.session_state.current_collection = sanitized_name
                            st.session_state.app_mode = "chat"
                            st.rerun()
    
    # Display existing collections
    st.markdown('<h2 class="section-header">🗄️ Your Knowledge Bases</h2>', unsafe_allow_html=True)
    
    # Load user collections
    user_collections = load_user_collections()
    
    if not user_collections:
        st.markdown('<div class="info-box">You have not created any knowledge bases yet. Upload documents above to get started.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"You have {len(user_collections)} knowledge base(s):")
        
        # Create a table for collections
        for collection_id, collection_data in user_collections.items():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{collection_data['display_name']}**")
                    st.markdown(f"<span class='collection-info'>Created: {collection_data['created_at']} | Files: {len(collection_data['files'])} | Chunks: {collection_data.get('chunks', 'N/A')}</span>", unsafe_allow_html=True)
                
                with col2:
                    if st.button("💬 Chat", key=f"chat_{collection_id}", help="Chat with this knowledge base"):
                        st.session_state.current_collection = collection_id
                        st.session_state.app_mode = "chat"
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{collection_id}", help="Delete this knowledge base", type="primary"):
                        if delete_collection(collection_id):
                            # Remove from user collections
                            del user_collections[collection_id]
                            save_user_collections(user_collections)
                            st.success(f"Knowledge base '{collection_data['display_name']}' deleted successfully.")
                            st.rerun()
                
                st.divider()

# Chat Mode
elif st.session_state.app_mode == "chat":
    st.markdown('<h2 class="section-header">💬 Chat with Your Documents</h2>', unsafe_allow_html=True)
    
    # Load user collections for dropdown
    user_collections = load_user_collections()
    
    if not user_collections:
        st.markdown('<div class="info-box">You have not created any knowledge bases yet. Please upload documents first.</div>', unsafe_allow_html=True)
        if st.button("📄 Go to Upload Documents"):
            st.session_state.app_mode = "upload"
            st.rerun()
    else:
        # Knowledge base selection
        collection_options = {collection_id: data["display_name"] for collection_id, data in user_collections.items()}
        
        # Default to current collection if set, otherwise first one
        default_index = 0
        if st.session_state.current_collection in collection_options:
            default_index = list(collection_options.keys()).index(st.session_state.current_collection)
        
        selected_collection_id = st.selectbox(
            "Select Knowledge Base:",
            options=list(collection_options.keys()),
            format_func=lambda x: collection_options[x],
            index=default_index
        )
        
        # Update current collection
        if st.session_state.current_collection != selected_collection_id:
            st.session_state.current_collection = selected_collection_id
            st.session_state.messages = []  # Clear chat when switching collections
        
        # Display collection info
        if selected_collection_id in user_collections:
            collection_data = user_collections[selected_collection_id]
            st.markdown(f'<div class="info-box">📚 Knowledge Base: <b>{collection_data["display_name"]}</b><br>Files: {len(collection_data["files"])} | Created: {collection_data["created_at"]}</div>', unsafe_allow_html=True)
        
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1} ({source.get('source', 'Unknown')}):**")
                            st.markdown(source["text"])
                            st.divider()
        
        # Query input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant documents
            with st.spinner("Searching for relevant information..."):
                try:
                    # Retrieve relevant chunks
                    sources = search_chunks(selected_collection_id, prompt)
                    
                    if not sources:
                        with st.chat_message("assistant"):
                            st.markdown("I couldn't find relevant information in your documents to answer this question. Please try a different question or upload more documents to your knowledge base.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "I couldn't find relevant information in your documents to answer this question. Please try a different question or upload more documents to your knowledge base."
                        })
                    else:
                        # Prepare context
                        context = "\n\n".join([source["text"] for source in sources])
                        
                        # Prepare prompt for LLM
                        full_prompt = f"""
                        Answer the following question based on the provided context from the documents.
                        
                        Context:
                        {context}
                        
                        Question: {prompt}
                        
                        Answer:
                        """
                        
                        # Get response from your custom LLM function
                        response = abc_response(full_prompt) if 'abc_response' in globals() else f"Based on the documents, I found information that relates to your question: '{prompt}'\n\nAfter analyzing the relevant sections, I can provide the following answer..."
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            
                            # Show sources
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1} ({source.get('source', 'Unknown')}):**")
                                    st.markdown(source["text"])
                                    st.divider()
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat controls
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear current chat history"):
                st.session_state.messages = []
                st.rerun()
        
        # Add button to go back to upload
        st.markdown("---")
        if st.button("📄 Upload More Documents", type="secondary"):
            st.session_state.app_mode = "upload"
            st.rerun()

# Help section in expandable container at the bottom
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### How to use DocuMind AI:
    
    1. **Upload Documents Mode**:
       - Create a new knowledge base by giving it a name
       - Upload one or more PDF documents
       - Process the documents to extract and index their content
       
    2. **Chat Mode**:
       - Select a knowledge base from the dropdown
       - Ask questions about your documents
       - View source information to see where the answers come from
       
    3. **Best Practices**:
       - More specific questions tend to get better answers
       - Each knowledge base should contain related documents
       - For large documents, consider splitting them into smaller parts
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;">
    <p style="color: #6B7280; font-size: 0.8rem;">DocuMind AI © 2025 | Powered by LangChain, Qdrant, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Function placeholder for LLM response (to be implemented)
def abc_response(prompt):
    """Placeholder for LLM function."""
    # This is where you would integrate your LLM
    return f"Based on the documents provided, I would answer that...\n\nNote: Replace this with your actual LLM integration."
