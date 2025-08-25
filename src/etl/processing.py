# src/etl/processing.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_text(text: str) -> str:
    """
    Applies basic cleaning to the extracted text.

    - Replaces multiple newline characters with a single space.
    - Replaces multiple spaces with a single space.
    - Strips leading/trailing whitespace.
    """
    # Replace multiple newlines and then multiple spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_data(processed_pages: list[dict]) -> list[dict]:
    """
    Cleans and chunks the text from processed pages, preserving metadata.

    Args:
        processed_pages: The list of page dictionaries from the ocr_agent.

    Returns:
        A list of dictionaries, where each dictionary represents a
        single chunk of text with its associated metadata.
    """
    # This text splitter is designed to keep paragraphs, sentences, and
    # words together as much as possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # The maximum size of a chunk (in characters)
        chunk_overlap=150,  # The overlap between chunks to maintain context
        length_function=len
    )

    all_chunks = []
    for page in processed_pages:
        # First, clean the raw text from the page
        cleaned_text = clean_text(page['text'])

        # Use the splitter to create chunks from the cleaned text
        chunks = text_splitter.split_text(cleaned_text)

        # Create a structured record for each chunk
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "page_number": page['page_number'],
                "content": chunk_text,
                "chunk_id": f"page_{page['page_number']}_chunk_{i}"
            })

    return all_chunks