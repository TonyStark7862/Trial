# Installation required
# pip install qdrant-client fastembed

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding

# Initialize client
client = QdrantClient(":memory:")  # Use in-memory storage for demo

# Sample documents about machine learning
documents = [
    "Feature scaling is crucial for distance-based algorithms like K-Nearest Neighbors.",
    "Deep learning models use neural networks with multiple hidden layers.",
    "Regularization techniques like L1/L2 help prevent model overfitting.",
    "Natural Language Processing deals with text analysis and generation.",
    "Clustering algorithms group similar data points without supervision."
]

# Initialize embedding models
dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

# Generate embeddings
dense_embeddings = list(dense_model.embed(documents))
sparse_embeddings = list(sparse_model.embed(documents))
colbert_embeddings = list(colbert_model.embed(documents))

# Create collection with hybrid configuration
client.create_collection(
    collection_name="ml-docs",
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
        payload={"text": doc}
    )
    for idx, (dense_embedding, sparse_embedding, colbert_embedding, doc) 
    in enumerate(zip(dense_embeddings, sparse_embeddings, colbert_embeddings, documents))
]

# Insert data
client.upsert(
    collection_name="ml-docs",
    points=points
)

# Test query about feature engineering
query = "How to prepare features for machine learning models?"

# Generate query embeddings
dense_query = next(dense_model.query_embed(query))
sparse_query = next(sparse_model.query_embed(query))
colbert_query = next(colbert_model.query_embed(query))

# Hybrid search with reranking
results = client.query_points(
    collection_name="ml-docs",
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

# Display results
print("Top 3 Reranked Results:")
for idx, hit in enumerate(results):
    print(f"{idx+1}. {hit.payload['text']}\nScore: {hit.score:.4f}")
