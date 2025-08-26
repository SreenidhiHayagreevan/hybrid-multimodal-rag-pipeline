# src/data_ingestion/ocr_agent.py (FINAL VERSION - with Google OCR)

import fitz # PyMuPDF
# Import our new, real OCR function from the extractors module
from .extractors import extract_text_with_ocr

def needs_ocr(page_text: str, char_threshold: int = 100) -> bool:
    """
    The agent's decision logic. It determines if a page likely needs OCR.
    
    Returns True if the text extracted by standard methods is shorter than the
    threshold, which is a strong indicator of a scanned or image-based page.
    """
    if len(page_text.strip()) < char_threshold:
        return True
    return False

def process_document(pdf_path: str) -> list[dict]:
    """
    Processes a document page by page, intelligently deciding whether to use
    standard text extraction or to call the commercial OCR tool. This function
    is the core of the Agentic OCR Framework.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return []

    processed_pages = []
    for page_num, page in enumerate(doc):
        page_number = page_num + 1 # Use 1-based page numbers for clarity
        
        # 1. Always attempt the fast, standard extraction first.
        standard_text = page.get_text()

        # 2. The agent makes its decision based on the result of the standard extraction.
        if needs_ocr(standard_text):
            print(f"  -> Page {page_number}: Decision - OCR required. Calling Google Vision AI...")
            # If OCR is needed, call our new, real OCR function for this specific page.
            ocr_text = extract_text_with_ocr(pdf_path, page_number)
            processed_pages.append({
                "page_number": page_number,
                "text": ocr_text,
                "extraction_method": "ocr"
            })
        else:
            # If the page is machine-readable, use the text we already extracted.
            # This saves time and money by avoiding unnecessary API calls.
            processed_pages.append({
                "page_number": page_number,
                "text": standard_text,
                "extraction_method": "standard"
            })

    doc.close()
    return processed_pages