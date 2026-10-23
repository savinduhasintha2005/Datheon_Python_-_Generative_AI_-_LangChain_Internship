# Custom Document RAG Question Answering System

A Python-based RAG application that answers questions from custom PDF and TXT documents using LangChain, ChromaDB, Hugging Face embeddings, and Google Gemini.

## Features
- Load PDF and TXT documents.
- Split text using RecursiveCharacterTextSplitter.
- Generate vector embeddings locally.
- Store embeddings in ChromaDB.
- Retrieve relevant context using similarity search.
- Generate answers using Google Gemini.
- Display source filenames and PDF page citations.

## Technologies
- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Google Gemini API
- PyPDF

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

3. Add PDF or TXT files to the `documents/` folder.

4. Run the application:

   ```bash
   python main.py
   ```

## Usage

- Select option `1` to load documents and build the vector database.
- Select option `2` to ask questions about the documents.
- Select option `3` to exit.

The application retrieves relevant document chunks and generates answers with source references.

## Project Structure

```text
document_rag_project/
├── main.py
├── requirements.txt
├── .env
├── documents/
└── chroma_db/
```

