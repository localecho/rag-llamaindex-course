"""
Module 3: Embeddings
====================

This module covers embeddings - the core technology that enables
semantic search in RAG systems.

Learning Objectives:
- Understand what embeddings are and how they work
- Learn about different embedding models
- Implement embeddings in LlamaIndex
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SECTION 1: What are Embeddings?
# =============================================================================

"""
Embeddings Explained:
--------------------

Embeddings are numerical representations (vectors) of text that capture
semantic meaning. Similar texts have similar embeddings.

Text → Embedding Model → Vector (list of numbers)

"I love programming" → [0.012, -0.234, 0.891, ..., 0.156]  # 1536 dims
"Coding is great"    → [0.015, -0.228, 0.885, ..., 0.149]  # Similar!
"I hate rainy days"  → [-0.512, 0.123, -0.234, ..., 0.789] # Different

Why Embeddings Work:
- Trained on massive text corpora
- Learn semantic relationships
- "King" - "Man" + "Woman" ≈ "Queen"
- Enable similarity search
"""


def explain_embeddings():
    """Explain what embeddings are."""

    print("=" * 60)
    print("WHAT ARE EMBEDDINGS?")
    print("=" * 60)

    print("""
Think of embeddings as a way to convert text into a "location" in
semantic space, where similar meanings are close together.

EXAMPLE - 2D Simplified View:
-----------------------------

           Programming ↑
                       │    • "Python tutorial"
                       │    • "JavaScript guide"
                       │
                       │          • "Machine learning"
                       │
    ←─────────────────┼───────────────────────→ Science
                       │
                       │    • "Cooking recipes"
                       │    • "Baking bread"
                       │
           Cooking     ↓

Real embeddings have 384-3072 dimensions, not 2!

HOW SIMILARITY WORKS:
---------------------

Vector A: [0.1, 0.2, 0.3]  "machine learning"
Vector B: [0.1, 0.2, 0.35] "deep learning"
Vector C: [0.9, 0.1, 0.1]  "cooking food"

Similarity(A, B) = 0.98  (very similar - both about ML)
Similarity(A, C) = 0.12  (very different topics)

Common similarity metrics:
- Cosine Similarity: Angle between vectors (most common)
- Dot Product: Magnitude-weighted similarity
- Euclidean Distance: Straight-line distance
""")


# =============================================================================
# SECTION 2: Embedding Models
# =============================================================================

def explain_embedding_models():
    """Explain different embedding models available."""

    print("\n" + "=" * 60)
    print("EMBEDDING MODELS")
    print("=" * 60)

    models = [
        {
            "name": "OpenAI text-embedding-3-small",
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost": "$0.02 / 1M tokens",
            "quality": "Excellent",
            "use_case": "Best balance of quality and cost"
        },
        {
            "name": "OpenAI text-embedding-3-large",
            "dimensions": 3072,
            "max_tokens": 8191,
            "cost": "$0.13 / 1M tokens",
            "quality": "Best",
            "use_case": "Maximum quality, higher cost"
        },
        {
            "name": "OpenAI text-embedding-ada-002",
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost": "$0.10 / 1M tokens",
            "quality": "Great",
            "use_case": "Legacy model, still widely used"
        },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "max_tokens": 256,
            "cost": "Free (local)",
            "quality": "Good",
            "use_case": "Fast, free, runs locally"
        },
        {
            "name": "BAAI/bge-large-en-v1.5",
            "dimensions": 1024,
            "max_tokens": 512,
            "cost": "Free (local)",
            "quality": "Excellent",
            "use_case": "High quality local embedding"
        },
        {
            "name": "Cohere embed-english-v3.0",
            "dimensions": 1024,
            "max_tokens": 512,
            "cost": "$0.10 / 1M tokens",
            "quality": "Excellent",
            "use_case": "Alternative to OpenAI"
        }
    ]

    print("Popular Embedding Models:\n")
    for model in models:
        print(f"{model['name']}")
        print(f"  Dimensions: {model['dimensions']}")
        print(f"  Max Tokens: {model['max_tokens']}")
        print(f"  Cost: {model['cost']}")
        print(f"  Quality: {model['quality']}")
        print(f"  Use Case: {model['use_case']}")
        print()


# =============================================================================
# SECTION 3: Using Embeddings in LlamaIndex
# =============================================================================

def demo_openai_embeddings():
    """Demonstrate OpenAI embeddings in LlamaIndex."""

    print("\n" + "=" * 60)
    print("OPENAI EMBEDDINGS DEMO")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("[Demo Mode - No API Key]")
        print("""
To run this demo with actual embeddings:
1. Get API key from https://platform.openai.com/api-keys
2. Add to .env file: OPENAI_API_KEY=your-key-here
""")
        show_embedding_code_example()
        return

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        print("[1] Creating OpenAI Embedding Model")
        print("-" * 40)

        # Initialize embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            # dimensions=1536,  # Optional: reduce dimensions
        )

        print("Model: text-embedding-3-small")
        print("Dimensions: 1536")

        print("\n[2] Generating Embeddings")
        print("-" * 40)

        # Single text embedding
        text = "LlamaIndex is a framework for building RAG applications."
        embedding = embed_model.get_text_embedding(text)

        print(f"Text: \"{text}\"")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")

        print("\n[3] Batch Embeddings")
        print("-" * 40)

        texts = [
            "Machine learning is a subset of AI.",
            "Python is a popular programming language.",
            "Deep learning uses neural networks.",
        ]

        embeddings = embed_model.get_text_embedding_batch(texts)

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            print(f"Text {i + 1}: \"{text[:40]}...\"")
            print(f"  Dimensions: {len(emb)}")

        print("\n[4] Computing Similarity")
        print("-" * 40)

        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        query = "What is artificial intelligence?"
        query_embedding = embed_model.get_text_embedding(query)

        print(f"Query: \"{query}\"\n")
        print("Similarity scores:")
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            similarity = cosine_similarity(query_embedding, emb)
            print(f"  [{similarity:.4f}] {text}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Install with: pip install llama-index-embeddings-openai")
    except Exception as e:
        print(f"Error: {e}")


def show_embedding_code_example():
    """Show embedding code without running it."""

    print("""
EMBEDDING CODE EXAMPLE:
-----------------------

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Option 1: Configure globally
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# Option 2: Use directly
embed_model = OpenAIEmbedding()

# Generate single embedding
text = "Hello, world!"
embedding = embed_model.get_text_embedding(text)
print(f"Dimensions: {len(embedding)}")  # 1536

# Generate batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = embed_model.get_text_embedding_batch(texts)

# Query embedding (for search)
query = "What is AI?"
query_embedding = embed_model.get_query_embedding(query)
""")


def demo_local_embeddings():
    """Demonstrate local embeddings with HuggingFace."""

    print("\n" + "=" * 60)
    print("LOCAL EMBEDDINGS (HuggingFace)")
    print("=" * 60)

    print("""
For cost-free, offline embeddings, use HuggingFace models:

INSTALLATION:
-------------
pip install llama-index-embeddings-huggingface
pip install sentence-transformers

USAGE:
------
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Use a local model (downloads automatically)
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Configure globally
Settings.embed_model = embed_model

# Use in indexing
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

POPULAR LOCAL MODELS:
--------------------
1. BAAI/bge-small-en-v1.5
   - Dimensions: 384
   - Good quality, fast

2. BAAI/bge-base-en-v1.5
   - Dimensions: 768
   - Better quality, moderate speed

3. BAAI/bge-large-en-v1.5
   - Dimensions: 1024
   - Best quality, slower

4. sentence-transformers/all-MiniLM-L6-v2
   - Dimensions: 384
   - Very fast, decent quality
""")


# =============================================================================
# SECTION 4: Similarity Search
# =============================================================================

def explain_similarity_search():
    """Explain how similarity search works."""

    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH")
    print("=" * 60)

    print("""
HOW SIMILARITY SEARCH WORKS IN RAG:
-----------------------------------

1. INDEXING PHASE:
   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
   │   Chunks    │ -> │  Embedding  │ -> │ Vector Store │
   │ "chunk 1"   │    │  [0.1, ...]  │    │   ID: 1      │
   │ "chunk 2"   │    │  [0.3, ...]  │    │   ID: 2      │
   │ "chunk 3"   │    │  [-0.2, ...] │    │   ID: 3      │
   └─────────────┘    └─────────────┘    └──────────────┘

2. QUERY PHASE:
   ┌─────────────┐    ┌─────────────┐
   │   Query     │ -> │  Embedding  │
   │ "What is X?"│    │  [0.15, ...] │
   └─────────────┘    └─────────────┘
           │
           ▼
   ┌──────────────────────────────────────────────────────┐
   │  Compare query embedding to all stored embeddings:   │
   │                                                      │
   │  Query vs Chunk 1: similarity = 0.92  ← Most similar │
   │  Query vs Chunk 2: similarity = 0.45                 │
   │  Query vs Chunk 3: similarity = 0.23                 │
   └──────────────────────────────────────────────────────┘
           │
           ▼
   Return top-k (e.g., top 3) most similar chunks

SIMILARITY METRICS:
-------------------

1. Cosine Similarity (most common for text):
   sim(A, B) = (A · B) / (||A|| × ||B||)
   Range: -1 to 1 (1 = identical)

2. Dot Product:
   sim(A, B) = A · B
   Range: -∞ to +∞ (higher = more similar)

3. Euclidean Distance:
   dist(A, B) = √(Σ(Ai - Bi)²)
   Range: 0 to +∞ (lower = more similar)

LlamaIndex uses cosine similarity by default.
""")


def demo_vector_store_retrieval():
    """Demonstrate vector store retrieval configuration."""

    print("\n" + "=" * 60)
    print("VECTOR STORE RETRIEVAL")
    print("=" * 60)

    print("""
CONFIGURING RETRIEVAL IN LLAMAINDEX:
------------------------------------

from llama_index.core import VectorStoreIndex

# Create index
index = VectorStoreIndex.from_documents(documents)

# Basic retriever (top 2 results)
retriever = index.as_retriever(similarity_top_k=2)
nodes = retriever.retrieve("What is machine learning?")

# Print retrieved nodes
for node in nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:100]}...")
    print()

# Query engine with retrieval config
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Retrieve top 5 chunks
    response_mode="compact"  # Combine context
)

RETRIEVAL PARAMETERS:
--------------------

similarity_top_k: int
    Number of similar chunks to retrieve (default: 2)
    Higher = more context, but slower and may include irrelevant info

response_mode: str
    - "refine": Iteratively refine answer with each chunk
    - "compact": Combine all chunks, answer once
    - "tree_summarize": Build summary tree
    - "simple_summarize": Simple concatenation

FILTERING RETRIEVAL:
-------------------
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="category", value="technical")]
)

retriever = index.as_retriever(
    similarity_top_k=5,
    filters=filters
)
""")


# =============================================================================
# SECTION 5: Embedding Best Practices
# =============================================================================

def embedding_best_practices():
    """Share best practices for embeddings."""

    print("\n" + "=" * 60)
    print("EMBEDDING BEST PRACTICES")
    print("=" * 60)

    print("""
1. CONSISTENCY IS KEY
   - Use the SAME embedding model for indexing and querying
   - Mixing models = broken similarity search

   WRONG:
   ✗ Index with text-embedding-ada-002
   ✗ Query with text-embedding-3-small

   RIGHT:
   ✓ Index with text-embedding-3-small
   ✓ Query with text-embedding-3-small

2. MATCH CHUNK SIZE TO MODEL
   - Most models have max token limits
   - Truncation loses information

   Model                    Max Tokens
   text-embedding-3-small   8191
   all-MiniLM-L6-v2        256
   bge-large-en-v1.5       512

3. NORMALIZE YOUR EMBEDDINGS
   - Many models already normalize
   - Normalized = unit length vectors
   - Enables dot product = cosine similarity

4. BATCH YOUR REQUESTS
   - Single API call for multiple texts
   - Much faster than one-by-one

   # Slow
   for text in texts:
       embed_model.get_text_embedding(text)

   # Fast
   embed_model.get_text_embedding_batch(texts)

5. CACHE EMBEDDINGS
   - Embeddings are deterministic
   - Same text = same embedding
   - Store in vector DB, don't regenerate

6. CONSIDER DIMENSIONALITY
   - Higher dims = more nuance, more storage
   - Lower dims = faster, less storage
   - text-embedding-3-* allows dimension reduction

7. TEST WITH YOUR DATA
   - Academic benchmarks ≠ your use case
   - Test retrieval quality on your documents
   - Measure: precision, recall, MRR
""")


# =============================================================================
# SECTION 6: Complete Example
# =============================================================================

def complete_embedding_example():
    """Show a complete embedding workflow."""

    print("\n" + "=" * 60)
    print("COMPLETE EMBEDDING WORKFLOW")
    print("=" * 60)

    print("""
FULL RAG PIPELINE WITH EMBEDDINGS:
----------------------------------

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. Configure embedding model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    # Optional: reduce dimensions for faster similarity
    # dimensions=512,
)

# 2. Configure LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# 3. Load documents
documents = SimpleDirectoryReader("./data").load_data()
print(f"Loaded {len(documents)} documents")

# 4. Create index (embeddings generated here)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True  # Shows embedding progress
)

# 5. Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Retrieve top 3 chunks
)

# 6. Query!
response = query_engine.query("What is the main topic?")
print(response)

# 7. Access source nodes
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Source: {node.metadata.get('file_name')}")
    print(f"Text: {node.text[:200]}...")
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 3: EMBEDDINGS")
    print("=" * 60)

    # Run all demos
    explain_embeddings()
    explain_embedding_models()
    demo_openai_embeddings()
    demo_local_embeddings()
    explain_similarity_search()
    demo_vector_store_retrieval()
    embedding_best_practices()
    complete_embedding_example()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Embeddings convert text to vectors for semantic search
2. Similar texts have similar embeddings (close in vector space)
3. OpenAI embeddings are high quality but cost money
4. Local models (HuggingFace) are free but require compute
5. Always use the SAME model for indexing and querying
6. Batch embedding requests for efficiency
7. similarity_top_k controls how many chunks to retrieve

Recommended Starting Point:
- OpenAI text-embedding-3-small for production
- BAAI/bge-small-en-v1.5 for local/free usage

Next: Capstone Project - End-to-End RAG Application
""")
