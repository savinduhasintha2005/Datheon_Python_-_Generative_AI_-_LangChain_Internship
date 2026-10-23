# Document AI Extractor

A Python application that extracts structured invoice
information from unstructured text using Google Gemini.

## Technologies

- Python
- Google Gemini API
- Google Gen AI SDK
- LangChain PromptTemplate
- Pydantic
- JSON

## Setup

1. Create a virtual environment.

   python -m venv .venv

2. Activate it on Windows.

   .venv\Scripts\activate

3. Install dependencies.

   pip install -r requirements.txt

4. Configure GEMINI_API_KEY in the .env file.

5. Add invoice text to sample_invoice.txt.

6. Run the application.

   python main.py

## Output

The application creates output.json after successful
extraction and validation.

## Validation

- Invoice date format
- Positive item quantities
- Non-negative amounts
- Invoice total consistency

## Security

Never commit your API key or confidential invoices.
