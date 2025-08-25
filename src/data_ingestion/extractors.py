# src/data_ingestion/extractors.py
import fitz  # This is the PyMuPDF library

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text and metadata from each page of a PDF file.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A list of dictionaries, where each dictionary represents a page
        and contains its page number and extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or reading PDF file: {pdf_path}. Error: {e}")
        return []

    pages_content = []
    for page_num, page in enumerate(doc):
        pages_content.append({
            "page_number": page_num + 1, # Page numbers are 1-based for humans
            "text": page.get_text(),
        })
    
    doc.close()
    return pages_content

# This is a placeholder for our commercial OCR solution for later
def extract_text_with_ocr(pdf_path: str) -> list[dict]:
    """Placeholder function for a commercial OCR solution."""
    print(f"--- SIMULATING OCR for {pdf_path} ---")
    # In a real implementation, you would convert PDF pages to images
    # and send them to an API like Google Cloud Vision.
    return []