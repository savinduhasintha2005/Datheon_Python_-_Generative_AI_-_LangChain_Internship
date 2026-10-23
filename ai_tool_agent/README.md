# Gemini AI Tool-Calling Agent

A Python AI agent built with **Google Gemini** and **LangChain** that uses tools to answer user questions.

## Features
- **Web Search** – Search the internet using DuckDuckGo.
- **Calculator** – Perform mathematical calculations.
- **File Reader** – Read local TXT, Markdown, and JSON files.
- **Agent Execution Trace** – Display tool actions and observations.

## Technologies
- Python 3
- LangChain
- Google Gemini API
- DuckDuckGo Search
- python-dotenv

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your Gemini API key to `.env`:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

3. Run the application:

   ```bash
   python main.py
   ```

## Project Structure

```text
ai_tool_agent/
├── main.py
├── agent.py
├── tools.py
├── requirements.txt
├── .env
└── data/
    └── project_notes.txt
```

