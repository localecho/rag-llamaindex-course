"""
Module 3: Data Loaders
======================

This module covers different data loaders in LlamaIndex for ingesting
documents from various sources.

Learning Objectives:
- Understand different data loader types
- Load data from files, directories, and URLs
- Handle different file formats (TXT, PDF, DOCX)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SECTION 1: Understanding Data Loaders
# =============================================================================

"""
Data Loaders in LlamaIndex:
---------------------------

Data loaders (also called "Readers") are responsible for:
1. Reading data from various sources
2. Converting data into Document objects
3. Extracting metadata from files

Common Data Loaders:
- SimpleDirectoryReader: Load files from a directory
- PDFReader: Load PDF files
- DocxReader: Load Word documents
- CSVReader: Load CSV files
- JSONReader: Load JSON files
- WebPageReader: Load web pages
- NotionPageReader: Load Notion pages
- SlackReader: Load Slack messages
"""


def explain_data_loaders():
    """Explain the different data loader options."""

    print("=" * 60)
    print("DATA LOADERS OVERVIEW")
    print("=" * 60)

    loaders = [
        {
            "name": "SimpleDirectoryReader",
            "use_case": "Load all files from a directory",
            "formats": "TXT, PDF, DOCX, CSV, and more",
            "example": "SimpleDirectoryReader('./data').load_data()"
        },
        {
            "name": "PDFReader",
            "use_case": "Load PDF documents",
            "formats": "PDF",
            "example": "PDFReader().load_data('document.pdf')"
        },
        {
            "name": "DocxReader",
            "use_case": "Load Microsoft Word documents",
            "formats": "DOCX",
            "example": "DocxReader().load_data('document.docx')"
        },
        {
            "name": "CSVReader",
            "use_case": "Load tabular data",
            "formats": "CSV",
            "example": "CSVReader().load_data('data.csv')"
        },
        {
            "name": "WebPageReader",
            "use_case": "Load web pages",
            "formats": "HTML",
            "example": "WebPageReader().load_data(['https://example.com'])"
        },
        {
            "name": "JSONReader",
            "use_case": "Load JSON data",
            "formats": "JSON",
            "example": "JSONReader().load_data('data.json')"
        }
    ]

    for loader in loaders:
        print(f"\n{loader['name']}")
        print(f"  Use Case: {loader['use_case']}")
        print(f"  Formats: {loader['formats']}")
        print(f"  Example: {loader['example']}")


# =============================================================================
# SECTION 2: SimpleDirectoryReader
# =============================================================================

def create_sample_files():
    """Create sample files for testing data loaders."""

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "sample_docs"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a text file
    txt_content = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Popular algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
"""
    (data_dir / "ml_intro.txt").write_text(txt_content.strip())

    # Create another text file
    txt_content2 = """
Deep Learning Fundamentals

Deep learning is a subset of machine learning based on artificial neural
networks with multiple layers (hence "deep").

Key Concepts:
- Neurons and Activation Functions
- Forward and Backward Propagation
- Loss Functions and Optimization
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and Attention Mechanisms

Deep learning has revolutionized:
- Computer Vision
- Natural Language Processing
- Speech Recognition
- Autonomous Vehicles
"""
    (data_dir / "deep_learning.txt").write_text(txt_content2.strip())

    print(f"Created sample files in: {data_dir}")
    return data_dir


def demo_simple_directory_reader():
    """Demonstrate SimpleDirectoryReader usage."""

    print("\n" + "=" * 60)
    print("SIMPLE DIRECTORY READER DEMO")
    print("=" * 60)

    # Create sample files
    data_dir = create_sample_files()

    try:
        from llama_index.core import SimpleDirectoryReader

        print("\n[1] Basic Usage - Load all files from directory")
        print("-" * 40)

        reader = SimpleDirectoryReader(input_dir=str(data_dir))
        documents = reader.load_data()

        print(f"Loaded {len(documents)} documents\n")

        for i, doc in enumerate(documents):
            print(f"Document {i + 1}:")
            print(f"  Filename: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"  Characters: {len(doc.text)}")
            print(f"  Preview: {doc.text[:100]}...")
            print()

        print("\n[2] Load specific file types")
        print("-" * 40)
        print("""
# Load only .txt files
reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".txt"]
)

# Load multiple types
reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".txt", ".pdf", ".docx"]
)
""")

        print("\n[3] Recursive directory loading")
        print("-" * 40)
        print("""
# Load from subdirectories too
reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True
)
""")

        print("\n[4] Custom file handling")
        print("-" * 40)
        print("""
# Custom file extractor mapping
from llama_index.readers.file import PDFReader

reader = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={
        ".pdf": PDFReader()
    }
)
""")

        return documents

    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return None


# =============================================================================
# SECTION 3: File-Specific Readers
# =============================================================================

def demo_pdf_reader():
    """Demonstrate PDF reader usage."""

    print("\n" + "=" * 60)
    print("PDF READER DEMO")
    print("=" * 60)

    print("""
PDF Reader Usage:
-----------------

from llama_index.readers.file import PDFReader

# Initialize reader
reader = PDFReader()

# Load a single PDF
documents = reader.load_data("path/to/document.pdf")

# Each page becomes a separate document by default
for doc in documents:
    print(f"Page: {doc.metadata.get('page_label')}")
    print(f"Content: {doc.text[:200]}...")

Advanced Options:
-----------------
# Return one document per PDF (not per page)
reader = PDFReader(return_full_document=True)

# The PDF reader extracts:
# - Text content
# - Metadata (title, author, creation date)
# - Page numbers
""")


def demo_docx_reader():
    """Demonstrate DOCX reader usage."""

    print("\n" + "=" * 60)
    print("DOCX READER DEMO")
    print("=" * 60)

    print("""
DOCX Reader Usage:
------------------

# Using SimpleDirectoryReader (auto-detects .docx)
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_files=["document.docx"]
)
documents = reader.load_data()

# Using dedicated DocxReader
from llama_index.readers.file import DocxReader

reader = DocxReader()
documents = reader.load_data("document.docx")

The DOCX reader extracts:
- Text content with formatting
- Headers and paragraphs
- Tables (as text)
- Metadata
""")


# =============================================================================
# SECTION 4: Web and API Readers
# =============================================================================

def demo_web_reader():
    """Demonstrate web page reader usage."""

    print("\n" + "=" * 60)
    print("WEB PAGE READER DEMO")
    print("=" * 60)

    print("""
Web Page Reader Usage:
----------------------

from llama_index.readers.web import SimpleWebPageReader

# Load single URL
reader = SimpleWebPageReader()
documents = reader.load_data(["https://example.com"])

# Load multiple URLs
urls = [
    "https://docs.python.org/3/tutorial/",
    "https://docs.python.org/3/library/",
]
documents = reader.load_data(urls)

# With HTML to text conversion
reader = SimpleWebPageReader(html_to_text=True)

Advanced: BeautifulSoup Reader
------------------------------
from llama_index.readers.web import BeautifulSoupWebReader

reader = BeautifulSoupWebReader()
documents = reader.load_data(
    urls=["https://example.com"],
    custom_hostname="example.com"
)
""")


# =============================================================================
# SECTION 5: Document Object Structure
# =============================================================================

def explain_document_structure():
    """Explain the Document object structure."""

    print("\n" + "=" * 60)
    print("DOCUMENT OBJECT STRUCTURE")
    print("=" * 60)

    print("""
When you load data, you get Document objects with this structure:

Document
├── doc_id: str           # Unique identifier
├── text: str             # The actual content
├── metadata: dict        # Additional information
│   ├── file_name
│   ├── file_path
│   ├── file_type
│   ├── creation_date
│   └── (custom fields)
├── embedding: List[float] # Vector (added during indexing)
└── relationships: dict    # Links to other documents

Example:
--------
document = Document(
    text="This is the content...",
    metadata={
        "file_name": "report.pdf",
        "author": "John Doe",
        "department": "Engineering"
    }
)

Accessing document properties:
-----------------------------
print(document.text)          # Get content
print(document.doc_id)        # Get ID
print(document.metadata)      # Get all metadata
print(document.get_content()) # Get formatted content
""")


# =============================================================================
# SECTION 6: Custom Metadata
# =============================================================================

def demo_custom_metadata():
    """Demonstrate adding custom metadata to documents."""

    print("\n" + "=" * 60)
    print("CUSTOM METADATA DEMO")
    print("=" * 60)

    print("""
Adding Custom Metadata:
-----------------------

# Method 1: During loading with file_metadata function
def get_metadata(file_path):
    return {
        "source": "internal_docs",
        "department": "engineering",
        "confidential": True
    }

reader = SimpleDirectoryReader(
    input_dir="./data",
    file_metadata=get_metadata
)

# Method 2: Modify after loading
documents = reader.load_data()
for doc in documents:
    doc.metadata["category"] = "technical"
    doc.metadata["version"] = "1.0"

# Method 3: Create documents manually
from llama_index.core import Document

doc = Document(
    text="Your content here",
    metadata={
        "source": "manual",
        "topic": "AI",
        "author": "Jane Smith"
    }
)

Why Metadata Matters:
--------------------
- Filter during retrieval
- Track document sources
- Add context for the LLM
- Enable document versioning
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 3: DATA LOADERS")
    print("=" * 60)

    # Run all demos
    explain_data_loaders()
    demo_simple_directory_reader()
    demo_pdf_reader()
    demo_docx_reader()
    demo_web_reader()
    explain_document_structure()
    demo_custom_metadata()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. SimpleDirectoryReader is the go-to for loading local files
2. Specialized readers exist for PDF, DOCX, Web, and more
3. Documents contain text, metadata, and optional embeddings
4. Custom metadata helps with filtering and context
5. LlamaHub has 100+ community data loaders

Useful Resources:
- LlamaHub: https://llamahub.ai/
- Documentation: https://docs.llamaindex.ai/

Next: Chunking and Tokenization
""")
