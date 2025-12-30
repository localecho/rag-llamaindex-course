"""
Module 1: Introduction to RAG Systems
=====================================

This module covers the fundamentals of Retrieval-Augmented Generation (RAG) systems,
explaining why they're important and how they work.

Learning Objectives:
- Understand the evolution of NLP and language models
- Learn why RAG is important for LLM applications
- Understand the components of a RAG system
"""

# =============================================================================
# SECTION 1: Why RAG is Important
# =============================================================================

"""
The Problem with Standard LLMs:
-------------------------------
1. Knowledge Cutoff: LLMs are trained on data up to a certain date
2. Hallucinations: LLMs can generate plausible but incorrect information
3. No Access to Private Data: LLMs don't know about your specific documents
4. Context Window Limits: Can't process entire document collections at once

RAG Solution:
-------------
RAG combines the power of:
- Retrieval: Finding relevant information from a knowledge base
- Generation: Using an LLM to synthesize coherent responses

This gives us:
- Up-to-date information
- Grounded responses based on actual documents
- Access to private/proprietary data
- Reduced hallucinations with source citations
"""


def explain_rag_importance():
    """Demonstrates why RAG is important with practical examples."""

    print("=" * 60)
    print("WHY RAG IS IMPORTANT")
    print("=" * 60)

    problems = {
        "Knowledge Cutoff": {
            "problem": "LLM trained in 2023 doesn't know about 2024 events",
            "rag_solution": "Retrieve current documents and use them as context"
        },
        "Hallucinations": {
            "problem": "LLM might make up facts that sound plausible",
            "rag_solution": "Ground responses in actual retrieved documents"
        },
        "Private Data": {
            "problem": "LLM doesn't know your company's internal docs",
            "rag_solution": "Index your documents and retrieve relevant ones"
        },
        "Context Limits": {
            "problem": "Can't fit 1000 pages into a single prompt",
            "rag_solution": "Retrieve only the most relevant chunks"
        }
    }

    for issue, details in problems.items():
        print(f"\n{issue}:")
        print(f"  Problem: {details['problem']}")
        print(f"  RAG Solution: {details['rag_solution']}")

    print("\n")


# =============================================================================
# SECTION 2: What is a RAG System
# =============================================================================

"""
RAG System Architecture:
------------------------

1. INDEXING PHASE (Offline/Batch):
   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
   │  Documents  │ -> │   Chunking   │ -> │  Embedding  │ -> │ Vector Store │
   └─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘

2. QUERY PHASE (Online/Real-time):
   ┌───────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────────┐
   │   Query   │ -> │  Embedding  │ -> │   Retrieval   │ -> │   Context    │
   └───────────┘    └─────────────┘    └───────────────┘    └──────────────┘
                                                                    │
                                                                    v
   ┌───────────┐    ┌─────────────┐    ┌───────────────────────────────────┐
   │  Response │ <- │     LLM     │ <- │  Query + Retrieved Context       │
   └───────────┘    └─────────────┘    └───────────────────────────────────┘
"""


def explain_rag_components():
    """Explains the core components of a RAG system."""

    print("=" * 60)
    print("RAG SYSTEM COMPONENTS")
    print("=" * 60)

    components = [
        {
            "name": "1. Document Loader",
            "purpose": "Load documents from various sources (PDF, TXT, Web, DB)",
            "example": "SimpleDirectoryReader, PDFReader, WebPageReader"
        },
        {
            "name": "2. Text Splitter/Chunker",
            "purpose": "Break documents into smaller, manageable chunks",
            "example": "SentenceSplitter, TokenTextSplitter"
        },
        {
            "name": "3. Embedding Model",
            "purpose": "Convert text chunks into vector representations",
            "example": "OpenAI text-embedding-ada-002, sentence-transformers"
        },
        {
            "name": "4. Vector Store",
            "purpose": "Store and index embeddings for fast retrieval",
            "example": "ChromaDB, Pinecone, Weaviate, FAISS"
        },
        {
            "name": "5. Retriever",
            "purpose": "Find most relevant chunks for a given query",
            "example": "VectorIndexRetriever, BM25Retriever"
        },
        {
            "name": "6. LLM (Language Model)",
            "purpose": "Generate responses using retrieved context",
            "example": "GPT-4, GPT-3.5-turbo, Claude, Llama"
        },
        {
            "name": "7. Response Synthesizer",
            "purpose": "Combine retrieved info into coherent response",
            "example": "CompactAndRefine, TreeSummarize"
        }
    ]

    for comp in components:
        print(f"\n{comp['name']}")
        print(f"  Purpose: {comp['purpose']}")
        print(f"  Examples: {comp['example']}")

    print("\n")


# =============================================================================
# SECTION 3: RAG Framework Overview
# =============================================================================

def explain_rag_workflow():
    """Explains the complete RAG workflow."""

    print("=" * 60)
    print("RAG WORKFLOW")
    print("=" * 60)

    print("""
PHASE 1: INDEXING (Prepare your knowledge base)
-----------------------------------------------
Step 1: Load Documents
        - Read files from directory, URLs, or databases

Step 2: Parse & Chunk
        - Split documents into smaller pieces
        - Typically 256-1024 tokens per chunk
        - Include overlap between chunks (e.g., 20 tokens)

Step 3: Generate Embeddings
        - Convert each chunk to a vector (e.g., 1536 dimensions)
        - Captures semantic meaning of the text

Step 4: Store in Vector Database
        - Index embeddings for fast similarity search
        - Store original text alongside vectors


PHASE 2: QUERYING (Answer user questions)
-----------------------------------------
Step 1: Receive Query
        - User asks a question

Step 2: Embed Query
        - Convert question to vector using same embedding model

Step 3: Retrieve Relevant Chunks
        - Find top-k most similar chunks using vector similarity
        - Common: cosine similarity, dot product

Step 4: Build Prompt
        - Combine query + retrieved chunks into LLM prompt
        - "Based on the following context, answer the question..."

Step 5: Generate Response
        - LLM produces answer grounded in retrieved context

Step 6: Return Response
        - Optionally include source citations
""")


# =============================================================================
# SECTION 4: Key Concepts Demo
# =============================================================================

def demonstrate_similarity_concept():
    """Demonstrates the concept of semantic similarity without actual embeddings."""

    print("=" * 60)
    print("SEMANTIC SIMILARITY CONCEPT")
    print("=" * 60)

    # Simulated example of how semantic similarity works
    documents = [
        "Python is a programming language known for its simple syntax.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather in Paris is usually mild in spring.",
        "Deep learning uses neural networks with many layers.",
        "JavaScript is commonly used for web development."
    ]

    queries_and_relevant = [
        {
            "query": "What programming languages are easy to learn?",
            "most_relevant": 0,  # Python document
            "explanation": "Query about easy programming → matches Python's 'simple syntax'"
        },
        {
            "query": "How does AI learn from data?",
            "most_relevant": 1,  # ML document
            "explanation": "Query about AI learning → matches machine learning description"
        },
        {
            "query": "What are neural networks?",
            "most_relevant": 3,  # Deep learning document
            "explanation": "Query about neural networks → matches deep learning description"
        }
    ]

    print("\nDocuments in our knowledge base:")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")

    print("\n\nSemantic Matching Examples:")
    for item in queries_and_relevant:
        print(f"\n  Query: '{item['query']}'")
        print(f"  Retrieved: [{item['most_relevant']}] {documents[item['most_relevant']]}")
        print(f"  Why: {item['explanation']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 1: INTRODUCTION TO RAG SYSTEMS")
    print("=" * 60 + "\n")

    # Run all explanations
    explain_rag_importance()
    explain_rag_components()
    explain_rag_workflow()
    demonstrate_similarity_concept()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. RAG combines retrieval + generation for better LLM responses
2. It solves knowledge cutoff, hallucination, and private data problems
3. Core components: Loader → Chunker → Embeddings → Vector Store → LLM
4. Two phases: Indexing (offline) and Querying (real-time)
5. Semantic similarity enables finding relevant context for any query

Next: Module 2 - Building Your First RAG System with LlamaIndex
""")
