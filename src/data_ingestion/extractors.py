# src/data_ingestion/extractors.py (FINAL VERSION - with Google OCR)

import fitz  # PyMuPDF
from google.cloud import vision
from pdf2image import convert_from_path # New library to handle PDF-to-image conversion
import io

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text and metadata from each page of a PDF file using standard,
    non-OCR methods provided by PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  -> ERROR: Could not open PDF '{pdf_path}' with PyMuPDF. Error: {e}")
        return []

    pages_content = []
    for page_num, page in enumerate(doc):
        pages_content.append({
            "page_number": page_num + 1,
            "text": page.get_text(),
        })
    doc.close()
    return pages_content

def extract_text_with_ocr(pdf_path: str, page_number: int) -> str:
    """
    Performs Optical Character Recognition (OCR) on a single page of a PDF
    using the Google Cloud Vision AI commercial service.
    
    Args:
        pdf_path: The path to the PDF file.
        page_number: The 1-based page number to perform OCR on.

    Returns:
        The extracted text from the page, or an empty string if an error occurs.
    """
    print(f"    -> Converting page {page_number} to image for OCR...")
    try:
        # 1. Convert the specific PDF page to an in-memory image object.
        #    `first_page` and `last_page` are 1-based, matching our page_number.
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if not images:
            print(f"  -> ERROR: pdf2image could not convert page {page_number}.")
            return ""

        # 2. Convert the image object (from PIL) into bytes that the API can read.
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        image_content = img_byte_arr.getvalue()

        # 3. Call the Google Cloud Vision API.
        #    The client automatically finds your credentials via the .env variable.
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_content)
        
        # We use `document_text_detection` as it is optimized for dense text
        # found in documents, which is more accurate than standard text detection.
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")

        print(f"    -> OCR for page {page_number} successful.")
        return response.full_text_annotation.text

    except Exception as e:
        print(f"  -> ERROR: An exception occurred during OCR on page {page_number}: {e}")
        return ""