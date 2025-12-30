# Building RAG Systems with LlamaIndex

A comprehensive implementation of RAG (Retrieval-Augmented Generation) systems based on the Analytics Vidhya course curriculum.

## Course Overview

This project covers building production-ready RAG systems using LlamaIndex, from fundamentals to advanced implementations.

## Project Structure

```
rag-llamaindex-course/
├── module_1_intro/           # Introduction to RAG Systems
│   └── 01_rag_fundamentals.py
├── module_2_first_rag/       # Getting Started with LlamaIndex
│   └── 01_first_rag_system.py
├── module_3_components/      # Components of LlamaIndex
│   ├── data_loaders/
│   │   └── 01_data_loaders.py
│   ├── chunking/
│   │   └── 01_chunking_tokenization.py
│   └── embeddings/
│       └── 01_embeddings.py
├── capstone/                 # End-to-End RAG Application
│   └── rag_application.py
├── data/
│   └── sample_docs/          # Sample documents for testing
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Get your OpenAI API key at: https://platform.openai.com/api-keys

## Modules

### Module 1: Introduction to RAG Systems
- Understanding why RAG is important
- What is a RAG system
- Overview of RAG Framework

### Module 2: Getting Started with LlamaIndex
- Introduction to LlamaIndex
- Components of LlamaIndex
- Build Your First RAG System

### Module 3: Components of LlamaIndex
- Data Loaders and implementation
- Chunking and Tokenization
- Node Parsers
- Embeddings and implementation

### Capstone Project
- End-to-end RAG application with:
  - Multiple data source ingestion
  - Custom chunking strategies
  - Vector store integration
  - Query engine with response synthesis

## Running the Examples

```bash
# Module 1 - RAG Fundamentals
python module_1_intro/01_rag_fundamentals.py

# Module 2 - First RAG System
python module_2_first_rag/01_first_rag_system.py

# Module 3 - Components
python module_3_components/data_loaders/01_data_loaders.py
python module_3_components/chunking/01_chunking_tokenization.py
python module_3_components/embeddings/01_embeddings.py

# Capstone - Full RAG Application
python capstone/rag_application.py
```

## Key Concepts Covered

1. **Data Ingestion** - Loading various data sources (PDF, TXT, DOCX)
2. **Indexing** - Organizing data for efficient retrieval
3. **Chunking** - Breaking documents into manageable pieces
4. **Embeddings** - Converting text to vector representations
5. **Query Engine** - Setting up retrieval mechanisms
6. **Response Synthesis** - Generating coherent answers

## Technologies Used

- **LlamaIndex** - LLM application framework
- **OpenAI** - LLM and embeddings provider
- **ChromaDB** - Vector database
- **Python 3.10+**

## License

MIT License - Feel free to use for learning and projects.
