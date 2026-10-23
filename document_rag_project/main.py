import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "custom_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"


def get_embeddings():
    """Create a local text embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def get_llm():
    """Create the Gemini chat model."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "Please configure it in your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0
    )


def load_documents():
    """
    Load PDF pages and TXT files.
    Each PDF page is a separate Document with
    source filename and 1-based page metadata.
    """

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    supported_files = (
        list(DOCUMENTS_DIR.glob("*.pdf"))
        + list(DOCUMENTS_DIR.glob("*.txt"))
    )

    if not supported_files:
        print("No PDF or TXT files found in documents/.")
        return documents

    for file_path in supported_files:

        print(f"Loading: {file_path.name}")

        try:
            if file_path.suffix.lower() == ".pdf":

                reader = PdfReader(str(file_path))

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):
                    text = page.extract_text() or ""
                    text = text.strip()

                    if not text:
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "page": page_number,
                                "file_type": "pdf"
                            }
                        )
                    )

            elif file_path.suffix.lower() == ".txt":

                text = file_path.read_text(
                    encoding="utf-8-sig"
                ).strip()

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "page": 1,
                                "file_type": "txt"
                            }
                        )
                    )

        except Exception as exc:
            print(
                f"Could not load {file_path.name}: {exc}"
            )

    print(f"Loaded {len(documents)} pages/documents.")
    return documents


def split_documents(documents):
    """Split loaded documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Created {len(chunks)} chunks.")

    return chunks


def build_vector_database():
    """
    Load documents, split them, generate embeddings,
    and persist the vectors in ChromaDB.

    Rebuilding replaces the existing collection.
    """

    documents = load_documents()

    if not documents:
        print(
            "No readable document text found. "
            "Add PDF/TXT files and try again."
        )
        return None

    chunks = split_documents(documents)

    if not chunks:
        print("No text chunks were created.")
        return None

    embeddings = get_embeddings()

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    vector_store.delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    print("\nVector database created successfully.")
    print(f"Stored {len(chunks)} chunks.")
    print(f"Database location: {CHROMA_DIR}")

    return vector_store


def get_vector_database():
    """Open the previously persisted Chroma collection."""

    if not CHROMA_DIR.exists():
        raise RuntimeError(
            "Vector database not found. "
            "Choose option 1 to index documents first."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    if vector_store._collection.count() == 0:
        raise RuntimeError(
            "The vector database is empty. "
            "Choose option 1 to index documents first."
        )

    return vector_store


def retrieve_context(vector_store, question):
    """
    Retrieve the top relevant chunks for a question.
    Returns documents with their original metadata.
    """

    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K
    )

    if not results:
        return []

    return [document for document, score in results]


def answer_question(question, vector_store=None):
    """
    Generate an answer strictly from retrieved context.
    Return source citations separately for transparency.
    """

    if not question.strip():
        print("Please enter a question.")
        return

    if vector_store is None:
        vector_store = get_vector_database()

    retrieved_docs = retrieve_context(
        vector_store,
        question
    )

    if not retrieved_docs:
        print(
            "\nAnswer: I could not find relevant information "
            "in the loaded documents."
        )
        return

    context_parts = []
    source_list = []

    for index, document in enumerate(
        retrieved_docs,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        source_label = f"S{index}"

        context_parts.append(
            f"[{source_label}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}"
        )

        source_list.append(
            {
                "label": source_label,
                "source": source,
                "page": page
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a question-answering assistant for custom documents.

STRICT RULES:
1. Answer using ONLY the supplied context.
2. Do not use outside knowledge or make up facts.
3. If the context does not contain enough evidence,
   say: "I could not find this information in the
   provided documents."
4. Cite supporting passages using their exact source
   labels, for example [S1] or [S2].
5. Never invent a filename, page number, quotation,
   statistic, or source label.
6. If the context gives only part of an answer,
   explain what is supported and what is missing.
7. Answer clearly in the same language as the question.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.content)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCE PAGES")
    print("=" * 60)

    seen = set()

    for source in source_list:
        citation = (
            source["source"],
            source["page"]
        )

        if citation in seen:
            continue

        seen.add(citation)

        print(
            f'[{source["label"]}] '
            f'{source["source"]}, '
            f'page {source["page"]}'
        )

    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("CUSTOM DOCUMENT RAG QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    vector_store = None

    while True:
        print("\nChoose an option:")
        print("1. Load documents and rebuild vector database")
        print("2. Ask a question")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        try:
            if choice == "1":
                vector_store = build_vector_database()

            elif choice == "2":
                question = input(
                    "\nEnter your question: "
                ).strip()

                answer_question(
                    question,
                    vector_store
                )

            elif choice == "3":
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Enter 1, 2, or 3.")

        except Exception as exc:
            print(f"\nApplication error: {exc}")


if __name__ == "__main__":
    main()
