# AI RAG Chat Application

A responsive conversational AI web application built with Streamlit,
Python, LangChain, and Google Gemini.

## Features

- Real-time streaming AI responses
- Chat history using Streamlit session state
- PDF and TXT document uploads
- RAG-based document retrieval with ChromaDB
- Gemini-powered conversational AI
- Clear chat and reset application controls
- Error handling for API and document processing failures

## Requirements

- Python 3.10 or newer
- Google Gemini API key
- Internet connection

## Installation

1. Create a virtual environment:

   python -m venv .venv

2. Activate it on Windows:

   .venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

4. Create a `.env` file and add:

   GOOGLE_API_KEY=your_gemini_api_key_here

5. Start the application:

   streamlit run app.py

6. Open the local URL displayed in your terminal,
   usually http://localhost:8501.

## Usage

1. Open the application in your browser.
2. Upload PDF or TXT files in the sidebar.
3. Click Process documents to build the knowledge index.
4. Ask questions in the chat input.
5. Watch responses stream in real time.

You can also chat without uploading documents.

## Technologies

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- Google Generative AI Embeddings
