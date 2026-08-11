# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

An AI-powered chatbot that combines Retrieval-Augmented Generation (RAG), a knowledge graph, long-term user memory, and tool-based retrieval within a LangGraph workflow.

## Overview

Traditional chatbots often struggle with:

- retaining user-specific information across conversations
- reasoning over relationships between entities
- combining static knowledge with dynamically retrieved information

This project addresses these challenges by combining multiple intelligence components into a single conversational system.

## Key Features

### 1. Hybrid RAG

The system retrieves relevant information from a local knowledge base using:

- text preprocessing and chunking
- sentence-transformer embeddings
- FAISS vector search
- retrieved context for answer generation

### 2. Knowledge Graph

The knowledge graph represents relationships between entities as nodes and edges.

It supports structured reasoning alongside semantic vector retrieval.

The implementation uses NetworkX for local graph storage.

### 3. Long-Term Memory

The system maintains user-specific information such as:

- preferences
- previous interactions
- conversational context

Memory is stored using Supabase/PostgreSQL.

### 4. LangGraph Orchestration

LangGraph coordinates the different stages of the chatbot pipeline, including:

- memory retrieval
- RAG retrieval
- knowledge graph processing
- tool usage
- response generation

### 5. Dynamic Tools

The chatbot can retrieve information from external sources using tools such as:

- Wikipedia
- arXiv

This allows the system to handle information beyond the static knowledge base.

### 6. Web Data Pipeline

The project includes a data pipeline for:

1. web scraping
2. text cleaning
3. preprocessing
4. chunking
5. embedding generation
6. vector indexing

## System Architecture

```text
                    User Query
                        |
                        v
                 FastAPI / UI
                        |
                        v
                  LangGraph
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
     Memory            RAG         Knowledge Graph
        |               |               |
        +---------------+---------------+
                        |
                        v
                  LLM Generation
                        |
                        v
                    Response

# Project Structure
memory-augmented-chatbot/
│
├── agent/              # LangGraph workflow and agent logic
├── api/                # FastAPI backend and web interface
├── assets/             # Project screenshots and examples
├── data_pipeline/      # Scraping, cleaning and embedding pipeline
├── evaluation/         # Evaluation scripts and results
├── knowledge_graph/    # Entity extraction and graph storage
├── memory/             # Chat history and user memory
├── rag/                # FAISS retrieval components
│
├── app.py              # Application entry point
├── app_streamlit.py    # Streamlit interface
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── CONCEPTS_GUIDE.md
├── QA_STUDY_GUIDE.md
└── README.md

# Technology Stack
| Component         | Technology                      |
| ----------------- | ------------------------------- |
| Programming       | Python                          |
| API               | FastAPI                         |
| LLM Orchestration | LangGraph                       |
| LLM               | Groq / LLaMA                    |
| Embeddings        | Sentence Transformers           |
| Vector Search     | FAISS                           |
| Knowledge Graph   | NetworkX                        |
| Database / Memory | Supabase PostgreSQL             |
| Web Scraping      | BeautifulSoup                   |
| UI                | HTML/CSS/JavaScript + Streamlit |

Setup
1. Clone the repository
git clone https://github.com/saanvib27/memory-augmented-chatbot.git
cd memory-augmented-chatbot

2. Create a virtual environment
python -m venv .venv
Windows:
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables
Windows:
copy .env.example .env
Then add the required API/database credentials to .env.
Do not commit .env to GitHub.

5. Run the application
For the Streamlit interface:
streamlit run app_streamlit.py

For the FastAPI backend:
uvicorn app:app --reload

Evaluation
The project includes an evaluation module for assessing retrieval and response quality using metrics related to:
context relevance
retrieval performance
response quality
faithfulness
Evaluation files are available in the evaluation/ directory.

