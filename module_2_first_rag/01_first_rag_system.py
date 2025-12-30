"""
Module 2: Building Your First RAG System with LlamaIndex
=========================================================

This module walks you through building a complete RAG system using LlamaIndex.
We'll cover the core components and build a working question-answering system.

Learning Objectives:
- Understand LlamaIndex architecture
- Build a basic RAG pipeline
- Query your indexed documents
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# SECTION 1: Introduction to LlamaIndex
# =============================================================================

"""
LlamaIndex (formerly GPT Index) is a data framework for LLM applications.

Key Features:
- Data Connectors: Load data from various sources
- Data Indexes: Structure data for efficient retrieval
- Query Interface: Natural language querying
- Integration: Works with various LLMs and vector stores

Core Abstractions:
- Documents: Raw text data
- Nodes: Chunks of documents
- Index: Data structure for retrieval
- Query Engine: Interface for asking questions
"""


# =============================================================================
# SECTION 2: Setting Up LlamaIndex
# =============================================================================

def check_api_key():
    """Verify OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("ERROR: Please set your OPENAI_API_KEY in .env file")
        print("Get your key at: https://platform.openai.com/api-keys")
        return False
    print("OpenAI API key found")
    return True


# =============================================================================
# SECTION 3: Building Your First RAG System
# =============================================================================

def create_sample_documents():
    """Create sample documents for our RAG system."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sample_docs"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Sample documents about a fictional company
    documents = {
        "company_overview.txt": """
TechVentures Inc. - Company Overview

TechVentures Inc. is a leading technology company founded in 2015 by Sarah Chen and
Michael Roberts. Headquartered in San Francisco, California, the company specializes
in artificial intelligence and machine learning solutions for enterprise clients.

Mission Statement:
"To democratize AI technology and make intelligent solutions accessible to businesses
of all sizes."

Company Values:
1. Innovation First - We constantly push boundaries
2. Customer Success - Our clients' success is our success
3. Ethical AI - We build responsible AI systems
4. Collaboration - Great things happen when we work together

The company has grown from a small startup of 5 employees to over 500 team members
across 10 global offices. TechVentures has raised $150 million in funding and serves
over 1,000 enterprise clients worldwide.
        """,

        "products.txt": """
TechVentures Products and Services

1. AI Analytics Platform
   - Real-time data analysis
   - Predictive modeling
   - Custom dashboards
   - Pricing: Starting at $999/month

2. SmartBot Pro
   - Conversational AI chatbot
   - Multi-language support
   - Integration with popular CRMs
   - 24/7 customer support automation
   - Pricing: Starting at $499/month

3. VisionAI
   - Computer vision solutions
   - Image and video analysis
   - Quality control automation
   - Pricing: Custom enterprise pricing

4. DataSync
   - Data integration platform
   - ETL automation
   - Real-time synchronization
   - Pricing: Starting at $299/month

Enterprise Support Tiers:
- Basic: Email support, 48-hour response
- Professional: Phone + Email, 24-hour response
- Enterprise: Dedicated account manager, 4-hour response
        """,

        "faq.txt": """
TechVentures Frequently Asked Questions

Q: How do I get started with TechVentures products?
A: You can sign up for a free 14-day trial on our website. Our onboarding team will
guide you through the setup process.

Q: What integrations do you support?
A: We support integrations with Salesforce, HubSpot, Slack, Microsoft Teams, AWS,
Google Cloud, and over 100 other platforms.

Q: Is my data secure?
A: Yes, we use enterprise-grade security including SOC 2 Type II certification,
end-to-end encryption, and GDPR compliance.

Q: Can I cancel my subscription anytime?
A: Yes, you can cancel your subscription at any time. We offer pro-rated refunds
for annual subscriptions.

Q: Do you offer custom solutions?
A: Yes, our enterprise team can build custom AI solutions tailored to your specific
business needs. Contact sales@techventures.com for more information.

Q: What is your uptime guarantee?
A: We guarantee 99.9% uptime for all our products, backed by our Service Level Agreement.
        """
    }

    # Write documents to files
    for filename, content in documents.items():
        filepath = data_dir / filename
        with open(filepath, "w") as f:
            f.write(content.strip())
        print(f"Created: {filepath}")

    return data_dir


def build_rag_system():
    """Build a complete RAG system with LlamaIndex."""

    print("\n" + "=" * 60)
    print("BUILDING YOUR FIRST RAG SYSTEM")
    print("=" * 60)

    # Step 1: Check API key
    print("\n[Step 1] Checking API key...")
    if not check_api_key():
        return None

    # Step 2: Create sample documents
    print("\n[Step 2] Creating sample documents...")
    data_dir = create_sample_documents()

    try:
        # Step 3: Import LlamaIndex components
        print("\n[Step 3] Importing LlamaIndex components...")
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            Settings,
        )
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Step 4: Configure LlamaIndex settings
        print("\n[Step 4] Configuring LlamaIndex...")
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Step 5: Load documents
        print("\n[Step 5] Loading documents...")
        reader = SimpleDirectoryReader(input_dir=str(data_dir))
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents")

        # Step 6: Create index
        print("\n[Step 6] Creating vector index...")
        print("(This embeds all documents and stores them for retrieval)")
        index = VectorStoreIndex.from_documents(documents)
        print("Index created successfully!")

        # Step 7: Create query engine
        print("\n[Step 7] Creating query engine...")
        query_engine = index.as_query_engine(
            similarity_top_k=3,  # Retrieve top 3 most relevant chunks
        )
        print("Query engine ready!")

        return query_engine

    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        return None


def interactive_query_session(query_engine):
    """Run an interactive query session."""

    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY SESSION")
    print("=" * 60)
    print("\nYou can now ask questions about TechVentures Inc.")
    print("Type 'quit' to exit.\n")

    sample_questions = [
        "Who founded TechVentures and when?",
        "What products does TechVentures offer?",
        "How much does SmartBot Pro cost?",
        "Is my data secure with TechVentures?",
        "What is the uptime guarantee?",
    ]

    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nSearching and generating response...")
        try:
            response = query_engine.query(query)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def demo_without_api():
    """Demonstrate the RAG workflow without making API calls."""

    print("\n" + "=" * 60)
    print("RAG WORKFLOW DEMONSTRATION (No API Required)")
    print("=" * 60)

    print("""
This is what happens when you run a RAG system:

1. DOCUMENT LOADING:
   ┌─────────────────────────────────────────────────────────┐
   │ SimpleDirectoryReader loads all .txt files from folder │
   │ Each file becomes a Document object with:              │
   │   - text: The content of the file                      │
   │   - metadata: filename, file path, etc.                │
   └─────────────────────────────────────────────────────────┘

2. INDEXING:
   ┌─────────────────────────────────────────────────────────┐
   │ VectorStoreIndex.from_documents(documents):            │
   │   a) Splits documents into chunks (nodes)              │
   │   b) Generates embeddings for each chunk               │
   │   c) Stores embeddings in vector store                 │
   └─────────────────────────────────────────────────────────┘

3. QUERYING:
   ┌─────────────────────────────────────────────────────────┐
   │ query_engine.query("What products are available?"):    │
   │   a) Embeds the query                                  │
   │   b) Finds similar chunks in vector store              │
   │   c) Retrieves top-k relevant chunks                   │
   │   d) Sends query + context to LLM                      │
   │   e) Returns generated response                        │
   └─────────────────────────────────────────────────────────┘
""")

    # Show sample code
    print("SAMPLE CODE:")
    print("-" * 40)
    print("""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index (embeddings generated automatically)
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Ask questions
response = query_engine.query("What is TechVentures?")
print(response)
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 2: BUILDING YOUR FIRST RAG SYSTEM")
    print("=" * 60)

    # Check if we should run in demo mode
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("\n[Demo Mode - No API Key Detected]")
        demo_without_api()
        print("\n" + "-" * 60)
        print("To run the full interactive RAG system:")
        print("1. Get your OpenAI API key from https://platform.openai.com/api-keys")
        print("2. Copy .env.example to .env and add your key")
        print("3. Run this script again")
    else:
        # Build and run the RAG system
        query_engine = build_rag_system()

        if query_engine:
            interactive_query_session(query_engine)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. LlamaIndex simplifies RAG with high-level abstractions
2. SimpleDirectoryReader loads documents from a folder
3. VectorStoreIndex handles chunking, embedding, and storage
4. Query engine provides a simple interface for Q&A
5. Just 4 lines of code to build a basic RAG system!

Next: Module 3 - Deep Dive into LlamaIndex Components
""")
