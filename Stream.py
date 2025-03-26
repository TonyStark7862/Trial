import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import PyPDF2
import uuid

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = f"pdf-docs-{str(uuid.uuid4())}"

# Streamlit interface
st.title("PDF Chat with Hybrid Search")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

def extract_text_with_page_numbers(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            # Store text with metadata
            all_texts.append({
                'text': text,
                'metadata': {
                    'file_name': pdf_file.name,
                    'page_number': page_num + 1
                }
            })
    return all_texts

def initialize_rag_system(documents):
    # Initialize client
    client = QdrantClient(":memory:")

    # Initialize embedding models
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    # Generate embeddings
    dense_embeddings = list(dense_model.embed([doc['text'] for doc in documents]))
    sparse_embeddings = list(sparse_model.embed([doc['text'] for doc in documents]))
    colbert_embeddings = list(colbert_model.embed([doc['text'] for doc in documents]))

    # Create collection with hybrid configuration
    client.create_collection(
        collection_name=st.session_state.collection_name,
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
            payload={
                "text": doc['text'],
                "metadata": doc['metadata']
            }
        )
        for idx, (dense_embedding, sparse_embedding, colbert_embedding, doc) 
        in enumerate(zip(dense_embeddings, sparse_embeddings, colbert_embeddings, documents))
    ]

    # Insert data
    client.upsert(
        collection_name=st.session_state.collection_name,
        points=points
    )

    return client, dense_model, sparse_model, colbert_model

def perform_hybrid_search(query, client, dense_model, sparse_model, colbert_model):
    # Generate query embeddings
    dense_query = next(dense_model.query_embed(query))
    sparse_query = next(sparse_model.query_embed(query))
    colbert_query = next(colbert_model.query_embed(query))

    # Hybrid search with reranking
    results = client.query_points(
        collection_name=st.session_state.collection_name,
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

# Process uploaded PDFs
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        documents = extract_text_with_page_numbers(uploaded_files)
        client, dense_model, sparse_model, colbert_model = initialize_rag_system(documents)
        st.session_state.rag_initialized = True
        st.success("PDFs processed successfully!")

# Chat interface
if 'rag_initialized' in st.session_state:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                st.write("Sources:")
                for source in message["sources"]:
                    st.write(f"- {source['file_name']}, Page {source['page_number']}")

    # Chat input
    user_query = st.chat_input("Ask a question about your PDFs")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get relevant documents
        results = perform_hybrid_search(user_query, client, dense_model, sparse_model, colbert_model)
        
        # Extract results using model_dump()
        points_list = results.model_dump()['points']
        extracted_data = [(point['payload']['text'], point['score']) for point in points_list]
        
        # Extract sources with scores
        sources = []
        for text, score in extracted_data:
            # Find the corresponding metadata
            for result in results:
                if result.payload['text'] == text:
                    sources.append({
                        'file_name': result.payload['metadata']['file_name'],
                        'page_number': result.payload['metadata']['page_number'],
                        'text': text,
                        'score': score
                    })
                    break

        # Generate context for LLM
        context = "\n".join([f"Content from {s['file_name']}, Page {s['page_number']}: {s['text']}" for s in sources])
        
        # Prepare prompt
        prompt = f"""Context: {context}\n\nQuestion: {user_query}\n\nPlease provide a detailed answer based on the context provided."""
        
        # Get LLM response
        response = abc_response(prompt)  # Using the external LLM function as requested

        # Display assistant response with sources
        with st.chat_message("assistant"):
            st.write(response)
            st.write("Sources:")
            for source in sources:
                st.write(f"- {source['file_name']}, Page {source['page_number']}")
                st.write(f"  Text: {source['text']}")
                st.write(f"  Score: {source['score']:.4f}")
                st.write("-" * 50)

        # Update chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
else:
    st.info("Please upload PDF files to start chatting.")
