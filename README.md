# AI PDF RAG Question Answering System

An AI-powered PDF Question Answering system using Retrieval-Augmented Generation (RAG).

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- Groq LLM
- NumPy

## Architecture

PDF Document
↓
Text Extraction
↓
Text Chunking
↓
Sentence Embeddings
↓
FAISS Vector Database
↓
User Question
↓
Semantic Similarity Search
↓
Relevant Document Chunks
↓
Groq LLM
↓
Grounded Answer

## Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate semantic embeddings
- Store embeddings in FAISS
- Perform semantic similarity search
- Generate grounded answers using Groq LLM
- Display retrieved sources and page numbers

## Installation

```bash
pip install -r Requirement.txt
