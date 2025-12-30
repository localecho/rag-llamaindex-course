"""
Module 3: Chunking and Tokenization
====================================

This module covers how to split documents into smaller chunks for
efficient retrieval and processing.

Learning Objectives:
- Understand why chunking is important
- Learn different chunking strategies
- Implement node parsers in LlamaIndex
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SECTION 1: Why Chunking Matters
# =============================================================================

"""
Why We Need Chunking:
---------------------

1. Context Window Limits
   - LLMs have token limits (e.g., 4K, 8K, 128K tokens)
   - Can't fit entire documents in a single prompt

2. Retrieval Precision
   - Smaller chunks = more precise retrieval
   - Find exactly the relevant part, not the whole document

3. Embedding Quality
   - Embeddings work better on focused content
   - Long texts dilute the semantic meaning

4. Cost Efficiency
   - Only process/send relevant chunks to LLM
   - Reduces API costs significantly

The Chunking Tradeoff:
----------------------
- Too Small: Loses context, fragments meaning
- Too Large: Dilutes relevance, wastes tokens
- Sweet Spot: Usually 256-1024 tokens with overlap
"""


def explain_chunking_importance():
    """Explain why chunking is critical for RAG systems."""

    print("=" * 60)
    print("WHY CHUNKING MATTERS")
    print("=" * 60)

    print("""
Imagine you have a 100-page document about a company.

User asks: "What is the CEO's email address?"

WITHOUT proper chunking:
------------------------
- Entire document goes to LLM (expensive!)
- LLM has to find needle in haystack
- May hit context window limits
- Retrieval matches entire document

WITH proper chunking:
---------------------
- Document split into ~200 chunks
- Query embedding matches chunk about CEO contact info
- Only relevant chunk sent to LLM
- Faster, cheaper, more accurate

CHUNK SIZE CONSIDERATIONS:
--------------------------
┌─────────────────────────────────────────────────────────────┐
│  Chunk Size    │  Pros                │  Cons              │
├─────────────────────────────────────────────────────────────┤
│  Small (128)   │  Precise retrieval   │  Loses context     │
│  Medium (512)  │  Good balance        │  Standard choice   │
│  Large (1024)  │  More context        │  Less precise      │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# SECTION 2: Tokenization Basics
# =============================================================================

def explain_tokenization():
    """Explain how tokenization works."""

    print("\n" + "=" * 60)
    print("TOKENIZATION BASICS")
    print("=" * 60)

    print("""
What is Tokenization?
---------------------
Converting text into tokens (smaller units) that models can process.

Example:
--------
Text: "Hello, how are you?"

Word tokens:  ["Hello", ",", "how", "are", "you", "?"]      (6 tokens)
BPE tokens:   ["Hello", ",", " how", " are", " you", "?"]   (6 tokens)
Character:    ["H","e","l","l","o",",","h","o","w"...]      (19 tokens)

Why It Matters for Chunking:
----------------------------
- LLMs count in TOKENS, not characters or words
- "Hello" = 1 token
- "Pneumonoultramicroscopicsilicovolcanoconiosis" = 10 tokens!
- Emojis and special characters can be multiple tokens

Token Counts (approximate):
---------------------------
- 1 token ≈ 4 characters (English)
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words
- 1000 tokens ≈ 750 words ≈ 1.5 pages
""")


def demo_tokenization():
    """Demonstrate tokenization with tiktoken."""

    print("\n" + "=" * 60)
    print("TOKENIZATION DEMO")
    print("=" * 60)

    try:
        import tiktoken

        # Get the tokenizer for GPT models
        encoding = tiktoken.get_encoding("cl100k_base")

        examples = [
            "Hello, world!",
            "Machine learning is fascinating.",
            "The quick brown fox jumps over the lazy dog.",
            "LlamaIndex helps build RAG applications.",
            "Email: support@example.com",
        ]

        print("\nTokenization Examples (cl100k_base encoding):\n")

        for text in examples:
            tokens = encoding.encode(text)
            print(f"Text: \"{text}\"")
            print(f"  Tokens: {tokens}")
            print(f"  Count: {len(tokens)} tokens")
            print(f"  Decoded: {[encoding.decode([t]) for t in tokens]}")
            print()

    except ImportError:
        print("tiktoken not installed. Install with: pip install tiktoken")
        print("\nShowing conceptual example instead:\n")
        print('Text: "Hello, world!"')
        print('  Tokens: [9906, 11, 1917, 0]')
        print('  Count: 4 tokens')
        print('  Decoded: ["Hello", ",", " world", "!"]')


# =============================================================================
# SECTION 3: Node Parsers in LlamaIndex
# =============================================================================

def explain_node_parsers():
    """Explain Node Parsers in LlamaIndex."""

    print("\n" + "=" * 60)
    print("NODE PARSERS IN LLAMAINDEX")
    print("=" * 60)

    print("""
What are Nodes?
---------------
- Nodes are chunks of documents in LlamaIndex
- Each node contains:
  - Text content
  - Metadata (inherited + chunk-specific)
  - Relationships (previous/next nodes)
  - Embedding (added during indexing)

Available Node Parsers:
-----------------------
1. SentenceSplitter
   - Splits on sentences while respecting token limits
   - Best for: General text documents

2. TokenTextSplitter
   - Splits based on token count
   - Best for: Precise token control

3. SemanticSplitterNodeParser
   - Uses embeddings to find natural break points
   - Best for: Documents with varying structure

4. CodeSplitter
   - Splits code while respecting syntax
   - Best for: Source code files

5. MarkdownNodeParser
   - Respects markdown structure (headers, sections)
   - Best for: Markdown documents

6. HTMLNodeParser
   - Parses HTML structure
   - Best for: Web pages
""")


def demo_sentence_splitter():
    """Demonstrate SentenceSplitter usage."""

    print("\n" + "=" * 60)
    print("SENTENCE SPLITTER DEMO")
    print("=" * 60)

    sample_text = """
Artificial Intelligence (AI) is transforming industries worldwide. From healthcare
to finance, AI applications are becoming increasingly sophisticated. Machine learning,
a subset of AI, enables computers to learn from data without explicit programming.

Deep learning, powered by neural networks, has achieved remarkable results in image
recognition, natural language processing, and game playing. Companies like OpenAI,
Google, and Meta are pushing the boundaries of what's possible with AI.

The future of AI holds both promise and challenges. While AI can automate tasks and
improve decision-making, it also raises ethical concerns about privacy, bias, and
job displacement. Responsible AI development is crucial for ensuring technology
benefits humanity.
"""

    try:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import Document

        print("Original text length:", len(sample_text), "characters")
        print("\n[1] Basic SentenceSplitter")
        print("-" * 40)

        # Create a document
        doc = Document(text=sample_text)

        # Basic sentence splitter
        splitter = SentenceSplitter(
            chunk_size=200,  # Target chunk size in tokens
            chunk_overlap=20  # Overlap between chunks
        )

        nodes = splitter.get_nodes_from_documents([doc])

        print(f"Created {len(nodes)} chunks:\n")
        for i, node in enumerate(nodes):
            print(f"Chunk {i + 1}:")
            print(f"  Length: {len(node.text)} characters")
            print(f"  Content: {node.text[:100]}...")
            print()

        print("\n[2] Configuring Chunk Size and Overlap")
        print("-" * 40)
        print("""
# Small chunks, high overlap (for precise retrieval)
splitter = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=32
)

# Large chunks, small overlap (for more context)
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=50
)

# Separator customization
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    paragraph_separator="\\n\\n",
    secondary_chunking_regex="[^,.;。]+[,.;。]?"
)
""")

    except ImportError:
        print("LlamaIndex not installed. Showing conceptual example:")
        print("""
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,    # Max tokens per chunk
    chunk_overlap=50   # Token overlap between chunks
)

nodes = splitter.get_nodes_from_documents(documents)
""")


def demo_token_text_splitter():
    """Demonstrate TokenTextSplitter usage."""

    print("\n" + "=" * 60)
    print("TOKEN TEXT SPLITTER DEMO")
    print("=" * 60)

    print("""
TokenTextSplitter Usage:
------------------------

from llama_index.core.node_parser import TokenTextSplitter

# Create splitter with token-based splitting
splitter = TokenTextSplitter(
    chunk_size=256,      # Tokens per chunk
    chunk_overlap=20,    # Token overlap
    separator=" ",       # Split on spaces
    backup_separators=["\\n"]
)

# Split documents
nodes = splitter.get_nodes_from_documents(documents)

Key Differences from SentenceSplitter:
--------------------------------------
- TokenTextSplitter: Strict token count, may split mid-sentence
- SentenceSplitter: Respects sentence boundaries, approximate token count

When to use TokenTextSplitter:
-----------------------------
- Need exact token control
- Working with non-sentence text (code, logs)
- Embedding model has strict token limits
""")


def demo_semantic_splitter():
    """Demonstrate SemanticSplitterNodeParser."""

    print("\n" + "=" * 60)
    print("SEMANTIC SPLITTER DEMO")
    print("=" * 60)

    print("""
SemanticSplitterNodeParser:
---------------------------

This advanced splitter uses embeddings to find natural break points
in text based on semantic similarity.

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# Initialize with embedding model
embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1,           # Sentences to group
    breakpoint_percentile_threshold=95,  # Similarity threshold
    embed_model=embed_model
)

How it works:
-------------
1. Splits text into sentences
2. Generates embeddings for each sentence
3. Compares adjacent sentence embeddings
4. Breaks where similarity drops significantly

Benefits:
---------
- Creates semantically coherent chunks
- Better for varied document structures
- Chunks naturally align with topic changes

Drawbacks:
----------
- Requires embedding API calls
- More expensive/slower
- Chunk sizes can vary significantly
""")


# =============================================================================
# SECTION 4: Overlap and Metadata
# =============================================================================

def explain_chunk_overlap():
    """Explain the importance of chunk overlap."""

    print("\n" + "=" * 60)
    print("CHUNK OVERLAP EXPLAINED")
    print("=" * 60)

    print("""
Why Use Chunk Overlap?
----------------------

Without overlap, information at chunk boundaries can be lost or split:

Document: "The CEO announced the merger on Monday. The deal is worth $5B."

Without Overlap:
┌────────────────────────┐  ┌────────────────────────┐
│ Chunk 1                │  │ Chunk 2                │
│ "The CEO announced     │  │ "The deal is worth     │
│  the merger on Monday."│  │  $5B."                 │
└────────────────────────┘  └────────────────────────┘

Query: "When was the merger announced?"
- Chunk 1 has context ✓
- But what if chunk boundary was different?

With Overlap (recommended):
┌────────────────────────────────┐
│ Chunk 1                        │
│ "The CEO announced the merger  │
│  on Monday. The deal is..."    │
└────────────────────────────────┘
        ┌────────────────────────────────┐
        │ Chunk 2                        │
        │ "...Monday. The deal is worth  │
        │  $5B."                         │
        └────────────────────────────────┘

Recommended Overlap:
-------------------
- chunk_overlap = 10-20% of chunk_size
- chunk_size=512 → chunk_overlap=50-100
- chunk_size=256 → chunk_overlap=25-50
""")


def explain_metadata_inheritance():
    """Explain how metadata flows to nodes."""

    print("\n" + "=" * 60)
    print("METADATA INHERITANCE")
    print("=" * 60)

    print("""
When documents are chunked, metadata flows to nodes:

Document Metadata:
{
    "file_name": "report.pdf",
    "author": "Jane Doe",
    "date": "2024-01-15"
}
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Node 1                    │  Node 2                      │
│  metadata: {               │  metadata: {                 │
│    "file_name": "...",     │    "file_name": "...",       │
│    "author": "Jane Doe",   │    "author": "Jane Doe",     │
│    "date": "2024-01-15",   │    "date": "2024-01-15",     │
│    "_node_type": "text",   │    "_node_type": "text",     │
│    "start_char_idx": 0,    │    "start_char_idx": 500,    │
│    "end_char_idx": 499     │    "end_char_idx": 999       │
│  }                         │  }                           │
└───────────────────────────────────────────────────────────┘

Additional Node Metadata:
------------------------
- _node_type: Type of node (text, image, etc.)
- start_char_idx: Start position in original document
- end_char_idx: End position in original document
- relationships: Links to prev/next nodes

Using Metadata for Filtering:
----------------------------
# Filter by author during retrieval
retriever = index.as_retriever(
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Jane Doe")]
    )
)
""")


# =============================================================================
# SECTION 5: Practical Implementation
# =============================================================================

def full_chunking_example():
    """Complete chunking implementation example."""

    print("\n" + "=" * 60)
    print("COMPLETE CHUNKING EXAMPLE")
    print("=" * 60)

    print("""
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# 1. Load documents
documents = SimpleDirectoryReader("./data").load_data()

# 2. Create node parser with custom settings
node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    paragraph_separator="\\n\\n\\n",
)

# 3. Option A: Manual parsing
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} nodes")

# 4. Option B: Use ingestion pipeline (recommended)
pipeline = IngestionPipeline(
    transformations=[
        node_parser,
        # Add more transformations like embeddings
    ]
)
nodes = pipeline.run(documents=documents)

# 5. Create index from nodes
index = VectorStoreIndex(nodes)

# 6. Or include in index creation
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[node_parser]
)
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 3: CHUNKING AND TOKENIZATION")
    print("=" * 60)

    # Run all demos
    explain_chunking_importance()
    explain_tokenization()
    demo_tokenization()
    explain_node_parsers()
    demo_sentence_splitter()
    demo_token_text_splitter()
    demo_semantic_splitter()
    explain_chunk_overlap()
    explain_metadata_inheritance()
    full_chunking_example()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Chunking is essential for efficient RAG systems
2. Tokens ≠ words; use tokenizers to count accurately
3. SentenceSplitter is best for most text documents
4. Overlap prevents information loss at boundaries
5. Metadata flows from documents to nodes

Common Settings:
- General text: chunk_size=512, overlap=50
- Dense documents: chunk_size=256, overlap=30
- Long-form content: chunk_size=1024, overlap=100

Next: Embeddings Implementation
""")
