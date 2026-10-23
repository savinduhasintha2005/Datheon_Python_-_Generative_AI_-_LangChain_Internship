from io import BytesIO
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def load_document(uploaded_file):
    """Convert an uploaded PDF or TXT file into LangChain documents."""

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        documents = [
            Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": uploaded_file.name,
                    "page": page_number,
                },
            )
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    elif filename.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="replace")
        documents = [
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name},
            )
        ]

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    documents = [
        doc for doc in documents if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No readable text found. Scanned PDFs may require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)


def create_vector_store(documents):
    """Create an in-memory Chroma vector store."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )


def retrieve_context(vector_store, question):
    """Retrieve the most relevant document chunks."""

    results = vector_store.similarity_search(
        question,
        k=4,
    )

    context_parts = []

    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        location = f"{source}, page {page}" if page else source

        context_parts.append(
            f"Source: {location}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)
