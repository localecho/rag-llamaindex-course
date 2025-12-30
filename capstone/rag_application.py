"""
Capstone Project: End-to-End RAG Application
=============================================

This is a complete, production-ready RAG application that combines
all the concepts from the course:
- Data loading from multiple sources
- Custom chunking strategies
- Embeddings with vector storage
- Query engine with response synthesis
- Conversation history support

Features:
- Multi-document ingestion
- Persistent vector storage (ChromaDB)
- Configurable retrieval
- Source citation
- Interactive chat interface
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class RAGConfig:
    """Configuration for the RAG application."""

    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536

    # LLM settings
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.1

    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Retrieval settings
    SIMILARITY_TOP_K = 3

    # Storage
    PERSIST_DIR = "./storage"


# =============================================================================
# RAG APPLICATION CLASS
# =============================================================================

class RAGApplication:
    """
    Complete RAG Application for document Q&A.

    This class provides:
    - Document ingestion from files
    - Vector index creation with persistence
    - Query interface with source citations
    - Conversation support
    """

    def __init__(self, config: RAGConfig = None):
        """Initialize the RAG application."""
        self.config = config or RAGConfig()
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self._setup_components()

    def _setup_components(self):
        """Set up LlamaIndex components."""
        try:
            from llama_index.core import Settings
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OPENAI_API_KEY not set")
                return

            # Configure LLM
            Settings.llm = OpenAI(
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE
            )

            # Configure embedding model
            Settings.embed_model = OpenAIEmbedding(
                model=self.config.EMBEDDING_MODEL
            )

            print(f"LLM: {self.config.LLM_MODEL}")
            print(f"Embedding: {self.config.EMBEDDING_MODEL}")

        except ImportError as e:
            print(f"Import error: {e}")
            print("Install dependencies: pip install -r requirements.txt")

    def ingest_documents(self, input_path: str, recursive: bool = True) -> int:
        """
        Ingest documents from a directory or file.

        Args:
            input_path: Path to directory or file
            recursive: Whether to search subdirectories

        Returns:
            Number of documents ingested
        """
        try:
            from llama_index.core import (
                SimpleDirectoryReader,
                VectorStoreIndex,
                StorageContext,
            )
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb

            print(f"\n{'='*50}")
            print("DOCUMENT INGESTION")
            print(f"{'='*50}")

            # Load documents
            print(f"\n[1] Loading documents from: {input_path}")
            path = Path(input_path)

            if path.is_file():
                reader = SimpleDirectoryReader(input_files=[str(path)])
            else:
                reader = SimpleDirectoryReader(
                    input_dir=str(path),
                    recursive=recursive,
                    required_exts=[".txt", ".pdf", ".docx", ".md"]
                )

            documents = reader.load_data()
            print(f"    Loaded {len(documents)} documents")

            # Display document info
            for doc in documents:
                filename = doc.metadata.get('file_name', 'unknown')
                chars = len(doc.text)
                print(f"    - {filename}: {chars:,} characters")

            # Create node parser (chunking)
            print(f"\n[2] Chunking documents...")
            print(f"    Chunk size: {self.config.CHUNK_SIZE} tokens")
            print(f"    Chunk overlap: {self.config.CHUNK_OVERLAP} tokens")

            node_parser = SentenceSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )

            # Create ChromaDB client
            print(f"\n[3] Setting up vector store...")
            persist_dir = Path(self.config.PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)

            chroma_client = chromadb.PersistentClient(path=str(persist_dir))
            chroma_collection = chroma_client.get_or_create_collection("rag_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            # Create index
            print(f"\n[4] Creating vector index...")
            print(f"    (Generating embeddings for all chunks)")

            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[node_parser],
                show_progress=True
            )

            print(f"\n[5] Index created and persisted to: {persist_dir}")

            # Create query engine
            self._create_engines()

            return len(documents)

        except Exception as e:
            print(f"Error during ingestion: {e}")
            raise

    def load_existing_index(self) -> bool:
        """
        Load an existing index from storage.

        Returns:
            True if index loaded successfully
        """
        try:
            from llama_index.core import StorageContext, load_index_from_storage
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb

            persist_dir = Path(self.config.PERSIST_DIR)

            if not persist_dir.exists():
                print("No existing index found")
                return False

            print(f"Loading index from: {persist_dir}")

            chroma_client = chromadb.PersistentClient(path=str(persist_dir))
            chroma_collection = chroma_client.get_or_create_collection("rag_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            self.index = VectorStoreIndex.from_vector_store(vector_store)

            self._create_engines()
            print("Index loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def _create_engines(self):
        """Create query and chat engines from the index."""
        if self.index is None:
            return

        # Query engine (single Q&A)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.SIMILARITY_TOP_K,
            response_mode="compact"
        )

        # Chat engine (with history)
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_question",
            similarity_top_k=self.config.SIMILARITY_TOP_K
        )

    def query(self, question: str, show_sources: bool = True) -> str:
        """
        Query the RAG system.

        Args:
            question: The question to ask
            show_sources: Whether to display source documents

        Returns:
            The generated response
        """
        if self.query_engine is None:
            return "Error: No index loaded. Run ingest_documents() first."

        print(f"\nQuestion: {question}")
        print("-" * 50)

        response = self.query_engine.query(question)

        print(f"\nAnswer:\n{response}\n")

        if show_sources and response.source_nodes:
            print("Sources:")
            print("-" * 50)
            for i, node in enumerate(response.source_nodes, 1):
                score = node.score if hasattr(node, 'score') else 'N/A'
                filename = node.metadata.get('file_name', 'unknown')
                preview = node.text[:150].replace('\n', ' ')
                print(f"\n[{i}] {filename} (score: {score:.4f})")
                print(f"    \"{preview}...\"")

        return str(response)

    def chat(self, message: str) -> str:
        """
        Chat with conversation history.

        Args:
            message: User message

        Returns:
            Assistant response
        """
        if self.chat_engine is None:
            return "Error: No index loaded. Run ingest_documents() first."

        response = self.chat_engine.chat(message)
        return str(response)

    def reset_chat(self):
        """Reset conversation history."""
        if self.chat_engine:
            self.chat_engine.reset()
            print("Chat history cleared.")


# =============================================================================
# SAMPLE DATA CREATION
# =============================================================================

def create_sample_knowledge_base():
    """Create sample documents for the capstone project."""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "knowledge_base"
    data_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "company_handbook.txt": """
ACME Corporation Employee Handbook
Version 2024.1

CHAPTER 1: COMPANY OVERVIEW

ACME Corporation was founded in 1985 by John Smith and Mary Johnson. Starting as a
small software consultancy, we have grown into a global technology leader with over
10,000 employees across 25 countries.

Our Mission: To innovate and deliver technology solutions that make the world more
connected and efficient.

Our Values:
- Innovation: We embrace new ideas and technologies
- Integrity: We act ethically in all our business dealings
- Inclusion: We celebrate diversity and foster belonging
- Impact: We measure success by the positive change we create

CHAPTER 2: WORK POLICIES

Working Hours:
Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday. We offer
flexible working arrangements including:
- Remote work (up to 3 days per week for eligible roles)
- Flexible start times (between 7:00 AM and 10:00 AM)
- Compressed workweek (subject to manager approval)

Time Off Policy:
- Annual Leave: 20 days per year (increases with tenure)
- Sick Leave: 10 days per year
- Personal Days: 3 days per year
- Parental Leave: 16 weeks paid leave for all new parents

CHAPTER 3: BENEFITS

Health Insurance:
We provide comprehensive health coverage for all full-time employees, including:
- Medical insurance (PPO and HMO options)
- Dental and vision coverage
- Mental health support and counseling
- Wellness program with gym membership subsidies

Retirement:
- 401(k) with 6% company match
- Vesting schedule: 100% after 2 years
- Financial planning assistance available

Professional Development:
- $5,000 annual learning budget
- Conference attendance (1-2 per year)
- Internal mentorship program
- Leadership development tracks
        """,

        "product_catalog.txt": """
ACME Corporation Product Catalog 2024

PRODUCT LINE: ENTERPRISE SOFTWARE

1. ACME CloudSuite
   Description: Comprehensive cloud management platform
   Features:
   - Multi-cloud orchestration (AWS, Azure, GCP)
   - Cost optimization and forecasting
   - Security compliance monitoring
   - Auto-scaling and load balancing
   Pricing: Starting at $2,500/month
   Target: Enterprise IT teams

2. ACME DataFlow
   Description: Real-time data pipeline solution
   Features:
   - Stream processing at scale
   - Built-in data transformations
   - Connector library (100+ integrations)
   - Visual pipeline designer
   Pricing: Starting at $1,000/month
   Target: Data engineering teams

3. ACME SecureVault
   Description: Enterprise security and identity management
   Features:
   - Single Sign-On (SSO)
   - Multi-factor authentication
   - Privileged access management
   - Compliance reporting (SOC2, HIPAA, GDPR)
   Pricing: $8 per user/month
   Target: Security teams

4. ACME Analytics Pro
   Description: Business intelligence and reporting platform
   Features:
   - Interactive dashboards
   - Natural language queries
   - Automated report generation
   - Embedded analytics API
   Pricing: Starting at $500/month
   Target: Business analysts

PRODUCT LINE: DEVELOPER TOOLS

5. ACME DevKit
   Description: Integrated development environment
   Features:
   - AI-powered code completion
   - Built-in debugging tools
   - Git integration
   - Plugin marketplace
   Pricing: Free tier available, Pro at $15/month
   Target: Software developers

6. ACME TestRunner
   Description: Automated testing platform
   Features:
   - Unit, integration, and E2E testing
   - Parallel test execution
   - Test analytics and reporting
   - CI/CD integration
   Pricing: Starting at $200/month
   Target: QA and development teams
        """,

        "technical_docs.txt": """
ACME CloudSuite Technical Documentation

ARCHITECTURE OVERVIEW

ACME CloudSuite uses a microservices architecture deployed on Kubernetes.

Core Components:
1. API Gateway: Handles all incoming requests, rate limiting, and authentication
2. Service Mesh: Manages inter-service communication using Istio
3. Data Layer: PostgreSQL for transactional data, Redis for caching
4. Message Queue: Apache Kafka for event streaming
5. Monitoring: Prometheus + Grafana for metrics, ELK stack for logging

INSTALLATION GUIDE

Prerequisites:
- Kubernetes cluster (1.25+)
- Helm 3.x
- kubectl configured
- Minimum 16GB RAM, 4 CPUs per node

Quick Start:
1. Add ACME Helm repository:
   helm repo add acme https://charts.acme.com
   helm repo update

2. Create namespace:
   kubectl create namespace acme-cloudsuite

3. Install CloudSuite:
   helm install cloudsuite acme/cloudsuite -n acme-cloudsuite

4. Verify installation:
   kubectl get pods -n acme-cloudsuite

API REFERENCE

Authentication:
All API requests require a Bearer token in the Authorization header.

POST /api/v1/auth/token
Request: {"client_id": "xxx", "client_secret": "xxx"}
Response: {"access_token": "xxx", "expires_in": 3600}

Resources API:
GET /api/v1/resources - List all cloud resources
POST /api/v1/resources - Create new resource
GET /api/v1/resources/{id} - Get resource details
DELETE /api/v1/resources/{id} - Delete resource

Cost API:
GET /api/v1/costs/summary - Get cost summary
GET /api/v1/costs/forecast - Get cost forecast
GET /api/v1/costs/recommendations - Get optimization recommendations

TROUBLESHOOTING

Common Issues:

Q: Pods stuck in Pending state
A: Check node resources with 'kubectl describe node'. Ensure sufficient CPU/memory.

Q: API returning 401 errors
A: Token may be expired. Generate new token using /auth/token endpoint.

Q: High latency on queries
A: Check Redis cache hit rate. Consider increasing cache TTL or cache size.

Q: Data sync delays
A: Verify Kafka consumer lag. Scale consumer replicas if needed.

Support Contact:
- Email: support@acme.com
- Portal: https://support.acme.com
- Emergency: +1-800-ACME-HELP
        """,

        "faq.txt": """
ACME Corporation Frequently Asked Questions

GENERAL QUESTIONS

Q: How do I contact ACME customer support?
A: You can reach our support team through multiple channels:
   - Email: support@acme.com
   - Phone: +1-800-555-ACME (available 24/7)
   - Live Chat: Available on our website
   - Support Portal: https://support.acme.com

Q: What are ACME's business hours?
A: Our main offices operate Monday-Friday, 9 AM - 6 PM local time.
   Technical support is available 24/7 for enterprise customers.

Q: Does ACME offer training for its products?
A: Yes! We offer several training options:
   - Self-paced online courses (free)
   - Instructor-led virtual training
   - On-site training for enterprise customers
   - Certification programs
   Visit learn.acme.com for more information.

PRODUCT QUESTIONS

Q: Can I try ACME products before purchasing?
A: Yes, most products offer a 14-day free trial. Enterprise products
   offer custom proof-of-concept engagements.

Q: How does pricing work for ACME CloudSuite?
A: CloudSuite uses consumption-based pricing. You pay for:
   - Number of managed resources
   - Data processed
   - API calls
   Contact sales@acme.com for a custom quote.

Q: Is my data secure with ACME products?
A: Absolutely. We maintain:
   - SOC 2 Type II certification
   - ISO 27001 certification
   - GDPR compliance
   - Data encryption at rest and in transit
   - Regular third-party security audits

Q: Can ACME products integrate with my existing tools?
A: Yes, we offer 100+ pre-built integrations including:
   - Salesforce, HubSpot, Zendesk
   - Slack, Microsoft Teams
   - Jira, GitHub, GitLab
   - AWS, Azure, Google Cloud
   Custom integrations available via our API.

BILLING QUESTIONS

Q: What payment methods do you accept?
A: We accept credit cards, wire transfers, and ACH payments.
   Annual contracts can be invoiced quarterly.

Q: Can I cancel my subscription?
A: Monthly subscriptions can be cancelled anytime.
   Annual subscriptions can be cancelled with 30 days notice
   (pro-rated refund may apply).

Q: Do you offer discounts?
A: Yes, we offer:
   - Annual payment discount (15%)
   - Startup program (50% off for qualifying startups)
   - Non-profit discount (25% off)
   - Volume discounts for large deployments
        """
    }

    for filename, content in documents.items():
        filepath = data_dir / filename
        filepath.write_text(content.strip())
        print(f"Created: {filepath}")

    return data_dir


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def run_interactive_session(app: RAGApplication):
    """Run an interactive Q&A session."""

    print("\n" + "=" * 60)
    print("INTERACTIVE RAG SESSION")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit    - Exit the session")
    print("  /chat    - Switch to chat mode (with history)")
    print("  /query   - Switch to query mode (no history)")
    print("  /reset   - Clear chat history")
    print("  /sources - Toggle source display")
    print("\nAsk anything about the knowledge base!\n")

    mode = "query"  # or "chat"
    show_sources = True

    while True:
        try:
            user_input = input(f"[{mode}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/chat":
            mode = "chat"
            print("Switched to chat mode (with conversation history)")
            continue
        elif user_input.lower() == "/query":
            mode = "query"
            print("Switched to query mode (no history)")
            continue
        elif user_input.lower() == "/reset":
            app.reset_chat()
            continue
        elif user_input.lower() == "/sources":
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}")
            continue

        # Process query
        if mode == "query":
            app.query(user_input, show_sources=show_sources)
        else:
            response = app.chat(user_input)
            print(f"\nAssistant: {response}\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the capstone project."""

    print("\n" + "=" * 60)
    print("CAPSTONE: END-TO-END RAG APPLICATION")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("\n[Demo Mode - No API Key Detected]")
        print("""
To run this application:
1. Get your OpenAI API key from https://platform.openai.com/api-keys
2. Create .env file in project root: OPENAI_API_KEY=your-key-here
3. Run this script again

This capstone project demonstrates:
- Multi-document ingestion
- Persistent vector storage with ChromaDB
- Configurable chunking and retrieval
- Query engine with source citations
- Chat engine with conversation history
- Interactive CLI interface
        """)
        return

    # Initialize application
    app = RAGApplication()

    # Check for existing index
    storage_path = Path(RAGConfig.PERSIST_DIR)
    if storage_path.exists():
        print("\nExisting index found.")
        choice = input("Load existing index? (y/n): ").strip().lower()
        if choice == 'y':
            if app.load_existing_index():
                run_interactive_session(app)
                return

    # Create sample data and ingest
    print("\nCreating sample knowledge base...")
    data_dir = create_sample_knowledge_base()

    print("\nIngesting documents...")
    app.ingest_documents(str(data_dir))

    # Run interactive session
    run_interactive_session(app)


if __name__ == "__main__":
    main()
