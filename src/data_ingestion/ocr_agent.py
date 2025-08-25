# src/data_ingestion/ocr_agent.py
import fitz # PyMuPDF
from .extractors import extract_text_with_ocr # Import our placeholder OCR function

def needs_ocr(page_text: str, char_threshold: int = 100) -> bool:
    """
    A simple agentic function to determine if a page likely needs OCR.

    Args:
        page_text: The text extracted from a PDF page using standard methods.
        char_threshold: The minimum number of characters for a page to be
                        considered machine-readable.

    Returns:
        True if the page is likely a scanned image or has garbled text.
        False otherwise.
    """
    if len(page_text.strip()) < char_threshold:
        # If the page has very little text, it's a strong indicator
        # that it's an image or a scanned document.
        return True

    # Add more sophisticated checks here in the future.
    # For example, you could check for a high ratio of non-alphanumeric
    # characters, or use a language detection library to see if the text
    # is coherent. For now, the character count is a robust starting point.

    return False

def process_document(pdf_path: str) -> list[dict]:
    """
    Processes a document page by page, intelligently deciding whether to use
    standard text extraction or to simulate calling an OCR tool.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of dictionaries, each containing the page number, extracted text,
        and the method used for extraction ('standard' or 'ocr').
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or reading PDF file: {pdf_path}. Error: {e}")
        return []

    processed_pages = []
    for page_num, page in enumerate(doc):
        # 1. Always try the fast, standard extraction first.
        standard_text = page.get_text()

        # 2. The "agent" makes a decision based on the result.
        if needs_ocr(standard_text):
            print(f"Page {page_num + 1}: Decision - OCR required. Simulating call.")
            # Here you would call your *real* OCR function. We'll use our placeholder.
            # In a real scenario, you'd pass the page object or an image of it.
            # For simplicity, we are just noting the decision.
            processed_pages.append({
                "page_number": page_num + 1,
                "text": "[[SCANNED_CONTENT_PLACEHOLDER]]", # Placeholder for OCR'd text
                "extraction_method": "ocr"
            })
        else:
            print(f"Page {page_num + 1}: Decision - Standard extraction is sufficient.")
            processed_pages.append({
                "page_number": page_num + 1,
                "text": standard_text,
                "extraction_method": "standard"
            })

    doc.close()
    return processed_pages