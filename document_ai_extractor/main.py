import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError
from schemas import InvoiceData, GEMINI_INVOICE_SCHEMA
from prompts import build_prompt


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def extract_invoice(document_text: str) -> InvoiceData:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError(
            "Please configure GEMINI_API_KEY in your .env file."
        )

    if not document_text.strip():
        raise ValueError("The document is empty.")

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(
        document_text=document_text,
        document_type="invoice",
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_INVOICE_SCHEMA,
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response. "
            "The request may have been blocked or interrupted."
        )

    invoice = InvoiceData.model_validate_json(response.text)

    return invoice


def main():
    input_path = BASE_DIR / "sample_invoice.txt"
    output_path = BASE_DIR / "output.json"

    try:
        document_text = input_path.read_text(encoding="utf-8")
        invoice = extract_invoice(document_text)
        json_output = invoice.model_dump_json(indent=2)
        output_path.write_text(
            json_output + "\n",
            encoding="utf-8",
        )

        print("\nInvoice extraction successful!\n")
        print(json_output)
        print(f"\nOutput saved to: {output_path}")

    except FileNotFoundError:
        print("Error: sample_invoice.txt was not found.")
        sys.exit(1)

    except ValidationError as exc:
        print("Validation error: extracted data is invalid.")
        print(exc)
        sys.exit(1)

    except ValueError as exc:
        print(f"Input or extraction error: {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"Application/API error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
