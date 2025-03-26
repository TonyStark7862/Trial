# Installation required
# pip install qdrant-client fastembed
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding

# Initialize client
client = QdrantClient(":memory:")  # Use in-memory storage for demo

# Sample documents with longer paragraphs about different topics
documents = [
    "Feature scaling is a crucial preprocessing step in machine learning that normalizes the range of features to improve model performance. Without proper scaling, algorithms that compute distances between data points, such as K-Nearest Neighbors or Support Vector Machines, may be dominated by features with larger scales. Common techniques include Min-Max scaling, which transforms features to a specific range (typically 0-1), and Standardization, which rescales features to have zero mean and unit variance. Feature scaling also accelerates gradient descent convergence in many optimization algorithms by creating a more uniform error surface to traverse.",
    
    "The Renaissance was a period of European cultural, artistic, political, and scientific rebirth that marked the transition from the Middle Ages to modernity. Spanning roughly from the 14th to the 17th century, it began in Florence, Italy before spreading throughout Europe. The period was characterized by renewed interest in classical antiquity, the development of linear perspective in painting, revolutionary scientific discoveries, and the advent of the printing press which democratized knowledge. Key figures like Leonardo da Vinci, Michelangelo, and Galileo Galilei embodied the Renaissance ideal of the 'Universal Man' with their diverse intellectual and artistic pursuits.",
    
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. Unlike classical physics, quantum mechanics incorporates principles such as wave-particle duality, quantum entanglement, and the uncertainty principle. These phenomena challenge our intuitive understanding of reality, as particles can exist in multiple states simultaneously (superposition) until measured, and entangled particles can instantaneously affect each other regardless of distance. The mathematical formulations of quantum mechanics, such as Schrödinger's wave equation, have enabled technological advances including lasers, transistors, and magnetic resonance imaging.",
    
    "Modern literary criticism emerged in the early 20th century with approaches like New Criticism, which advocated close reading of texts with minimal consideration of historical or biographical context. This contrasted with earlier biographical approaches to literature. As the century progressed, diverse theoretical frameworks developed, including structuralism, which analyzes underlying patterns in texts; post-structuralism, which questions stable meanings; feminist criticism, exploring gender implications; and postcolonial criticism, examining literature through the lens of imperial relationships. Contemporary literary criticism often integrates multiple approaches, acknowledging that texts can be interpreted through various cultural, historical, and theoretical perspectives.",
    
    "Immunotherapy represents a revolutionary approach to cancer treatment that leverages the body's own immune system to identify and attack cancer cells. Unlike traditional treatments such as chemotherapy, which targets all rapidly dividing cells, immunotherapy specifically enhances immune responses against cancer. Major categories include checkpoint inhibitors, which block proteins that prevent immune cells from attacking cancer; CAR T-cell therapy, where a patient's T cells are modified to better recognize cancer cells; cancer vaccines, which stimulate immune responses to tumor-specific antigens; and monoclonal antibodies, which mark cancer cells for destruction. While immunotherapy has shown remarkable success in treating certain cancers, researchers continue working to expand its efficacy across more cancer types and reduce immune-related adverse events."
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
        payload={"text": doc, "doc_id": idx}
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

# Get individual search results to analyze source scores
dense_results = client.search(
    collection_name="ml-docs",
    query_vector=("miniLM", dense_query),
    limit=5,
    with_payload=True,
    with_vectors=False
)

sparse_results = client.search(
    collection_name="ml-docs",
    query_vector=("bm25", models.SparseVector(**sparse_query.as_object())),
    limit=5,
    with_payload=True,
    with_vectors=False
)

# Hybrid search with reranking
final_results = client.query_points(
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

# Create a mapping of document IDs to their scores in different models
dense_scores = {hit.id: hit.score for hit in dense_results}
sparse_scores = {hit.id: hit.score for hit in sparse_results}

# Display results in standard Qdrant format
print("\nSearch Results:")
print("-" * 50)
print(f"Query: {query}")
print("-" * 50)

for idx, hit in enumerate(final_results):
    doc_id = hit.payload['doc_id']
    
    # Get source scores
    dense_score = dense_scores.get(doc_id, None)
    sparse_score = sparse_scores.get(doc_id, None)
    
    # Print result in Qdrant standard format
    print(f"\n{idx+1}. Score: {hit.score:.4f}")
    print(f"   Text: {hit.payload['text'][:150]}...")
    
    # Print source scores in the standard format used in Qdrant examples
    print(f"   Source scores:")
    print(f"     dense: {dense_score:.4f}" if dense_score is not None else "     dense: not in top 5")
    print(f"     sparse: {sparse_score:.4f}" if sparse_score is not None else "     sparse: not in top 5")
    print(f"     colbert: {hit.score:.4f} (final)")
    print("-" * 50)
