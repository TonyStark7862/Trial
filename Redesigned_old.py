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
import logging
import json
import time
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_pdf_chat")

# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant instead

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
COLLECTIONS_INFO_PATH = VECTORDB_DIR / "collections_info.json"

# Page configuration and styling
st.set_page_config(
    page_title="Enterprise PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI
st.markdown("""
<style>
    /* Main App Styling */
    .main {
        background-color: #fafafa;
    }
    
    /* Header and title styling */
    .main-header {
        background-color: #3B82F6;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Cards styling */
    .stcard {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    
    /* PDF uploader styling */
    .upload-section {
        border: 2px dashed #3B82F6;
        border-radius: 0a.5rem;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem 0.5rem 0.5rem 0;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem 0.5rem 0 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Sources panel styling */
    .sources-panel {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
    }
    
    /* Status indicators */
    .status-success {
        color: #10B981;
        font-weight: 500;
    }
    
    .status-error {
        color: #EF4444;
        font-weight: 500;
    }
    
    .status-processing {
        color: #F59E0B;
        font-weight: 500;
    }
    
    /* Loading animation */
    .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3B82F6;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Collection dropdown styling */
    .stSelectbox label {
        font-weight: 500;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 0.25rem;
        padding: 0.75rem;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 0.5rem;
    }
    
    /* Sidebar adjustments */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Cards for instructions */
    .info-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Timeline for chat */
    .chat-timeline {
        position: relative;
        margin-left: 20px;
    }
    
    .chat-timeline:before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 2px;
        background: #E5E7EB;
    }
    
    /* PDF Document cards */
    .pdf-card {
        border: 1px solid #E5E7EB;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        background-color: white;
    }
    
    .pdf-card:hover {
        background-color: #F3F4F6;
    }
    
    /* Collection badge */
    .collection-badge {
        background-color: #3B82F6;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Statistics counter */
    .stat-counter {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3B82F6;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Horizontal divider */
    .divider {
        height: 1px;
        background-color: #E5E7EB;
        margin: 1rem 0;
    }
    
    /* Improve expander styling */
    .streamlit-expanderHeader {
        font-size: 0.875rem;
        font-weight: 500;
        color: #3B82F6;
    }
    
    /* Filter search box */
    .search-box {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load collections info
def load_collections_info():
    if COLLECTIONS_INFO_PATH.exists():
        try:
            with open(COLLECTIONS_INFO_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading collections info: {e}")
            return {}
    return {}

# Save collections info
def save_collections_info(collections_info):
    try:
        with open(COLLECTIONS_INFO_PATH, "w") as f:
            json.dump(collections_info, f)
    except Exception as e:
        logger.error(f"Error saving collections info: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if USE_FAISS and "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if not USE_FAISS and "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Load collections from saved file
collections_info = load_collections_info()

# Header section
st.markdown('<div class="main-header"><h1>🔍 Enterprise PDF Chat</h1><p>Upload PDF documents and get AI-powered answers from their content</p></div>', unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.markdown("## 📁 Document Management")
    
    # Collection management section
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("### Create New Collection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_collection_name = st.text_input("Collection Name", placeholder="Enter a name for your collection")
    with col2:
        add_collection = st.button("Create")
    
    if add_collection and new_collection_name:
        collection_id = f"collection_{uuid.uuid4().hex[:8]}"
        collections_info[collection_id] = {
            "name": new_collection_name,
            "created_at": datetime.now().isoformat(),
            "documents": [],
            "document_count": 0
        }
        save_collections_info(collections_info)
        st.success(f"✅ Collection '{new_collection_name}' created!")
        # Reload the page to show the new collection
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show existing collections
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("### Your Collections")
    
    if not collections_info:
        st.info("No collections yet. Create one to get started!")
    else:
        # Sort collections by creation date (newest first)
        sorted_collections = sorted(
            collections_info.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        collection_options = ["-- Select a Collection --"] + [
            f"{collection['name']} ({collection.get('document_count', 0)} docs)" 
            for _, collection in sorted_collections
        ]
        collection_ids = [""] + [collection_id for collection_id, _ in sorted_collections]
        
        selected_index = st.selectbox(
            "Select a collection to work with:", 
            options=range(len(collection_options)),
            format_func=lambda i: collection_options[i]
        )
        
        if selected_index > 0:
            selected_collection_id = collection_ids[selected_index]
            st.session_state.active_collection = selected_collection_id
            
            # Show stats for selected collection
            collection = collections_info[selected_collection_id]
            st.markdown(f"""
            <div style="margin-top: 10px; padding: 10px; background-color: #F3F4F6; border-radius: 5px;">
                <div class="stat-counter">{collection.get('document_count', 0)}</div>
                <div class="stat-label">Documents</div>
                <div class="divider"></div>
                <div class="stat-label">Created: {datetime.fromisoformat(collection['created_at']).strftime('%b %d, %Y')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PDF uploader
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("### Upload PDF Documents")
    
    if st.session_state.active_collection:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="PDFs will be added to the selected collection"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Calculate file hash for caching/collection
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                
                # Check if this file is already processed
                collection = collections_info[st.session_state.active_collection]
                file_exists = any(doc["file_hash"] == file_hash for doc in collection.get("documents", []))
                
                if not file_exists:
                    # Add file to collection
                    collection["documents"].append({
                        "filename": uploaded_file.name,
                        "file_hash": file_hash,
                        "added_at": datetime.now().isoformat(),
                        "size_kb": round(len(file_bytes) / 1024, 1)
                    })
                    collection["document_count"] = len(collection["documents"])
                    
                    # Save updated collections info
                    save_collections_info(collections_info)
                    
                    # Process the file for vector storage
                    if USE_FAISS:
                        # FAISS processing logic here
                        cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
                        
                        if not cache_file.exists():
                            with st.spinner(f"Processing {uploaded_file.name} with FAISS..."):
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
                                    
                                    # Create embeddings and vector store
                                    embeddings = HuggingFaceEmbeddings(
                                        model_name="sentence-transformers/all-mpnet-base-v2"
                                    )
                                    
                                    vector_store = FAISS.from_texts(chunks, embeddings)
                                    
                                    # Save to cache
                                    with open(cache_file, "wb") as f:
                                        pickle.dump(vector_store, f)
                                    
                                    logger.info(f"Processed and cached PDF: {uploaded_file.name}")
                                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing PDF with FAISS: {e}")
                                    st.error(f"Error processing {uploaded_file.name}: {e}")
                        else:
                            st.success(f"✅ {uploaded_file.name} already in cache, using existing embeddings")
                    else:
                        # Qdrant processing logic here
                        collection_name = f"{st.session_state.active_collection}_{file_hash}"
                        
                        with st.spinner(f"Processing {uploaded_file.name} with Qdrant..."):
                            # Setup Qdrant client and model
                            qdrant_client = setup_qdrant_client()
                            model = load_sentence_transformer()
                            
                            if qdrant_client:
                                try:
                                    # Create collection with appropriate vector size
                                    vector_size = model.get_sentence_embedding_dimension()
                                    create_collection(qdrant_client, collection_name, vector_size)
                                    
                                    # Process PDF and add to collection
                                    process_pdf_qdrant(file_bytes, collection_name)
                                    
                                    logger.info(f"Processed PDF with Qdrant: {uploaded_file.name}")
                                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing PDF with Qdrant: {e}")
                                    st.error(f"Error processing {uploaded_file.name}: {e}")
                            else:
                                st.error("Failed to initialize Qdrant client")
                else:
                    st.info(f"📄 {uploaded_file.name} is already in this collection")
            
            # After processing all files, refresh the page to update stats
            time.sleep(1)
            st.rerun()
    else:
        st.info("Please select or create a collection first")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear chat button
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("### Chat Controls")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.retriever = None if USE_FAISS else None
        st.session_state.collection_name = None if not USE_FAISS else None
        st.success("Chat history cleared!")
        time.sleep(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown("""
    This application uses:
    - Streamlit for UI
    - LangChain for text processing
    - Qdrant/FAISS for vector storage
    - Local LLM for responses
    
    Designed for enterprise document question-answering.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize the embedding model for Qdrant
@st.cache_resource
def load_sentence_transformer():
    """Initialize the embedding model from local path or download if not present."""
    try:
        logger.info("Loading embedding model from local path...")
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        logger.info("Model loaded from local path")
        return model
    except Exception as e:
        logger.warning(f"Model not found locally or error loading: {e}")
        logger.info("Downloading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save the model for future use
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save(LOCAL_MODEL_PATH)
        logger.info("Model downloaded and saved locally")
        return model

# Setup Qdrant client
@st.cache_resource
def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    try:
        logger.info("Setting up Qdrant client...")
        client = QdrantClient(path=str(VECTORDB_DIR / "qdrant_db"))
        logger.info("Qdrant client setup successful")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        return None

# Create collection for the PDF if it doesn't exist (Qdrant)
def create_collection(client, collection_name, vector_size=384):
    """Create a new collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created")
        else:
            logger.info(f"Using existing collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise e

# Process PDF and add to Qdrant
def process_pdf_qdrant(file_bytes, collection_name):
    try:
        logger.info(f"Processing PDF for collection: {collection_name}")
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
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Create embeddings for chunks
        logger.info("Creating embeddings for chunks...")
        embeddings = model.encode(chunks)
        
        # Check if collection exists and has points
        collection_info = qdrant_client.get_collection(collection_name)
        existing_count = collection_info.points_count
        
        # Skip if chunks are already added
        if existing_count > 0:
            logger.info(f"Document chunks already added to collection (found {existing_count} points)")
            return
        
        # Prepare points for upload
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": chunk, "chunk_id": idx}
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        
        # Upload to collection
        logger.info(f"Uploading {len(points)} points to Qdrant collection")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} chunks to collection")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise e

# Search for relevant chunks in Qdrant
def search_chunks(collection_name, query, limit=4):
    """Search for chunks similar to the query."""
    try:
        logger.info(f"Searching in collection: {collection_name} for query: {query}")
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query])[0]
        
        # Search in collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        logger.info(f"Found {len(search_results)} matching chunks")
        return [result.payload["text"] for result in search_results]
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        return []

# Main application area - showing documents in collection
if st.session_state.active_collection:
    collection = collections_info[st.session_state.active_collection]
    
    # Documents listing section
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown(f"## 📚 Documents in \"{collection['name']}\"")
    
    if not collection.get("documents", []):
        st.info("No documents in this collection yet. Use the sidebar to upload PDFs.")
    else:
        # Document search/filter
        doc_search = st.text_input("🔍 Filter documents", placeholder="Type to search documents")
        
        # Display documents in collection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("#### Document Name")
        with col2:
            st.markdown("#### Added On")
        with col3:
            st.markdown("#### Size")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        for doc in collection.get("documents", []):
            # Apply filter if search term is provided
            if doc_search and doc_search.lower() not in doc["filename"].lower():
                continue
                
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"📄 {doc['filename']}")
            with col2:
                added_date = datetime.fromisoformat(doc["added_at"]).strftime("%b %d, %Y")
                st.markdown(f"{added_date}")
            with col3:
                st.markdown(f"{doc['size_kb']} KB")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("## 💬 Chat with Documents")
    
    # Display chat messages
    if not st.session_state.messages:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 How to use this chat
        - Ask questions about the documents in the selected collection
        - Be specific to get accurate answers
        - View sources used to generate the response by expanding the sources panel
        - Clear chat history using the button in the sidebar
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        message_role = message["role"]
        message_class = "user-message" if message_role == "user" else "assistant-message"
        
        st.markdown(f'<div class="{message_class}">', unsafe_allow_html=True)
        st.markdown(f"**{message_role.capitalize()}**: {message['content']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📝 View Sources"):
                st.markdown('<div class="sources-panel">', unsafe_allow_html=True)
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(source)
                    if i < len(message["sources"]) - 1:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a loading state placeholder
    chat_loading = st.empty()
    
    # Chat input
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    prompt = st.chat_input("Ask a question about your documents...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show loading state
        with chat_loading.container():
            st.markdown('<div class="status-processing">Processing your query...</div>', unsafe_allow_html=True)
        
        if USE_FAISS:
            # FAISS implementation for retrieval
            # Load all retrievers for this collection
            if st.session_state.active_collection:
                collection = collections_info[st.session_state.active_collection]
                all_documents = collection.get("documents", [])
                
                if not all_documents:
                    with st.chat_message("assistant"):
                        st.markdown("Please upload PDF files to this collection first.")
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload PDF files to this collection first."})
                else:
                    # Get all file hashes in this collection
                    file_hashes = [doc["file_hash"] for doc in all_documents]
                    
                    # Load all retrievers
                    retrievers = []
                    for file_hash in file_hashes:
                        cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
                        if cache_file.exists():
                            try:
                                with open(cache_file, "rb") as f:
                                    vector_store = pickle.load(f)
                                retrievers.append(vector_store.as_retriever(search_kwargs={"k": 2}))
                            except Exception as e:
                                logger.error(f"Error loading cache file {file_hash}: {e}")
                    
                    if not retrievers:
                        with st.chat_message("assistant"):
                            st.markdown("No valid document embeddings found. Please try uploading the documents again.")
                        st.session_state.messages.append({"role": "assistant", "content": "No valid document embeddings found. Please try uploading the documents again."})
                    else:
                        # Get relevant documents from all retrievers
                        try:
                            all_docs = []
                            for retriever in retrievers:
                                docs = retriever.get_relevant_documents(prompt)
                                all_docs.extend(docs)
                            
                            # Sort by relevance (assuming most relevant come first from each retriever)
                            # Limit to top results
                            top_docs = all_docs[:4]
                            sources = [doc.page_content for doc in top_docs]
                            
                            # Prepare context
                            context = "\n\n".join(sources)
                            
                            # Prepare prompt for LLM
                            full_prompt = f"""
                            Answer the following question based on the provided context.
                            
                            Context:
                            {context}
                            
                            Question: {prompt}
                            
                            Answer:
                            """
                            
                            # Get response from your custom LLM function
                            try:
                                response = abc_response(full_prompt)  # Your custom LLM function
                            except NameError:
                                # Fallback if abc_response is not defined
                                logger.warning("abc_response function not defined, using fallback response")
                                response = f"Based on the documents in your collection, I found relevant information that would help answer your question about '{prompt}'."
                            
                            # Clear loading state
                            chat_loading.empty()
                            
                            # Display assistant response
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                
                                # Show sources
                                with st.expander("📝 View Sources"):
                                    st.markdown('<div class="sources-panel">', unsafe_allow_html=True)
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**Source {i+1}:**")
                                        st.markdown(source)
                                        if i < len(sources) - 1:
                                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add assistant message to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "sources": sources
                            })
                            
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            error_message = f"Error generating response: {str(e)}"
                            
                            # Clear loading state
                            chat_loading.empty()
                            
                            with st.chat_message("assistant"):
                                st.markdown(f'<div class="status-error">{error_message}</div>', unsafe_allow_html=True)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            # Qdrant implementation for retrieval
            if st.session_state.active_collection:
                collection = collections_info[st.session_state.active_collection]
                all_documents = collection.get("documents", [])
                
                if not all_documents:
                    # Clear loading state
                    chat_loading.empty()
                    
                    with st.chat_message("assistant"):
                        st.markdown("Please upload PDF files to this collection first.")
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload PDF files to this collection first."})
                else:
                    try:
                        # Get all file hashes in this collection
                        all_results = []
                        
                        for doc in all_documents:
                            file_hash = doc["file_hash"]
                            collection_name = f"{st.session_state.active_collection}_{file_hash}"
                            
                            # Search in each document's collection
                            try:
                                doc_results = search_chunks(collection_name, prompt, limit=2)
                                all_results.extend(doc_results)
                            except Exception as e:
                                logger.warning(f"Error searching in collection {collection_name}: {e}")
                                continue
                        
                        # Clear loading state
                        chat_loading.empty()
                        
                        if not all_results:
                            with st.chat_message("assistant"):
                                st.markdown("I couldn't find relevant information in the documents to answer your question. Please try a different question or upload more relevant documents.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "I couldn't find relevant information in the documents to answer your question. Please try a different question or upload more relevant documents."
                            })
                        else:
                            # Limit to top results
                            top_results = all_results[:4]
                            
                            # Prepare context
                            context = "\n\n".join(top_results)
                            
                            # Prepare prompt for LLM
                            full_prompt = f"""
                            Answer the following question based on the provided context.
                            
                            Context:
                            {context}
                            
                            Question: {prompt}
                            
                            Answer:
                            """
                            
                            # Get response from your custom LLM function
                            try:
                                response = abc_response(full_prompt)  # Your custom LLM function
                            except NameError:
                                # Fallback if abc_response is not defined
                                logger.warning("abc_response function not defined, using fallback response")
                                response = f"Based on the documents in your collection, I found relevant information that would help answer your question about '{prompt}'."
                            
                            # Display assistant response
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                
                                # Show sources
                                with st.expander("📝 View Sources"):
                                    st.markdown('<div class="sources-panel">', unsafe_allow_html=True)
                                    for i, source in enumerate(top_results):
                                        st.markdown(f"**Source {i+1}:**")
                                        st.markdown(source)
                                        if i < len(top_results) - 1:
                                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add assistant message to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "sources": top_results
                            })
                    
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        error_message = f"Error generating response: {str(e)}"
                        
                        # Clear loading state
                        chat_loading.empty()
                        
                        with st.chat_message("assistant"):
                            st.markdown(f'<div class="status-error">{error_message}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Welcome screen when no collection is selected
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.markdown("## 👋 Welcome to Enterprise PDF Chat")
    st.markdown("""
    This application allows you to chat with your PDF documents using AI. Get started by:
    
    1. Creating a collection in the sidebar
    2. Uploading one or more PDF documents
    3. Asking questions about the content of your documents
    
    The AI will find relevant information in your documents and provide answers with source references.
    """)
    
    # Sample questions panel
    st.markdown("### Example Questions You Can Ask")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - "What are the key findings in this report?"
        - "Summarize the main points of the document"
        - "What does the document say about [specific topic]?"
        """)
        
    with col2:
        st.markdown("""
        - "Compare the information across these documents"
        - "What are the recommendations mentioned?"
        - "Find all references to [specific term]"
        """)
    
    st.markdown("Please select or create a collection in the sidebar to get started.")
    st.markdown('</div>', unsafe_allow_html=True)

# Custom function to simulate LLM response (to be replaced with actual LLM integration)
def abc_response(prompt):
    """Placeholder for actual LLM integration. This would be replaced with the real LLM call."""
    # This is where you would integrate with your actual LLM
    # For now, we'll just return a placeholder response
    logger.info("Generating response using abc_response function")
    
    question = prompt.split("Question: ")[-1].split("\n")[0].strip()
    
    return f"Here's information based on the documents: The answer to '{question}' can be found in the relevant sections I've extracted from your documents. The documents provide detailed information on this topic, which I've summarized in my response based on the most relevant passages."
