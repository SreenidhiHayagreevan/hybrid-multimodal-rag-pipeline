# src/indexing/indexer.py (FINAL BATCH VERSION)

import os
import sys
import glob # Used to find all files matching a pattern
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # Provides a smart progress bar for loops

# Add the project's root directory to the Python path
# This ensures that we can import from other folders in the `src` directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import our custom-built modules for document processing and ETL
from src.data_ingestion.ocr_agent import process_document
from src.etl.processing import chunk_data

def get_db_connection():
    """Establishes and returns a direct connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            user="postgres",
            password="Sree@1997", # Your database password
            dbname="rag_db"
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"\nFATAL: Could not connect to the database. Please ensure the Docker container is running.")
        print(f"Error details: {e}")
        sys.exit(1) # Exit the script if the database connection fails

def run_indexing_for_file(pdf_path: str, model: SentenceTransformer) -> int:
    """
    Runs the full ingestion, ETL, and indexing pipeline for a single PDF file.

    Args:
        pdf_path: The path to the PDF file to be indexed.
        model: The pre-loaded SentenceTransformer model.

    Returns:
        The number of chunks successfully indexed from the document.
    """
    # Use the basename for cleaner logging
    file_name = os.path.basename(pdf_path)
    
    # 1. Ingestion: Use our agent to extract text, deciding on OCR if necessary.
    processed_pages = process_document(pdf_path)
    if not processed_pages:
        print(f"  -> No content could be extracted from {file_name}. Skipping.")
        return 0

    # 2. ETL: Clean and chunk the extracted text.
    chunks = chunk_data(processed_pages)
    if not chunks:
        print(f"  -> Content from {file_name} could not be split into chunks. Skipping.")
        return 0

    # 3. Embedding: Generate vector embeddings for the content of each chunk.
    chunk_contents = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(chunk_contents, show_progress_bar=False, normalize_embeddings=True)

    # 4. Indexing: Store the chunks and their embeddings in the database.
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()
    
    try:
        for i, chunk in enumerate(chunks):
            embedding_list = embeddings[i].tolist()
            # Create a globally unique chunk_id by prefixing it with the filename
            unique_chunk_id = f"{file_name}_{chunk['chunk_id']}"
            
            cursor.execute(
                """
                INSERT INTO chunks (page_number, chunk_id, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
                """,
                (chunk['page_number'], unique_chunk_id, chunk['content'], embedding_list)
            )
        conn.commit()
    except Exception as e:
        print(f"  -> An error occurred during database indexing for {file_name}: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
    
    return len(chunks)

# This is the main block that runs when you execute the script.
if __name__ == "__main__":
    dataset_path = "data/SampleDataSet/"
    
    # Use glob to find all files ending with .pdf in the directory and any subdirectories.
    pdf_files = glob.glob(os.path.join(dataset_path, "**/*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{dataset_path}'. Please ensure the dataset is in the correct location.")
    else:
        print("--- 🚀 Starting Full Batch Indexing Pipeline ---")
        print(f"Found {len(pdf_files)} PDF files to index.")
        
        # Load the embedding model once at the start to avoid reloading it for every file.
        print("Loading embedding model (this may take a moment)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded.")
        
        total_chunks_indexed = 0
        
        # Use tqdm to create a progress bar that wraps around our list of files.
        for pdf_file in tqdm(pdf_files, desc="Indexing Documents"):
            chunks_count = run_indexing_for_file(pdf_file, embedding_model)
            total_chunks_indexed += chunks_count
            
        print("\n--- ✅ Batch Pipeline Finished ---")
        print(f"📊 Total chunks indexed from all documents: {total_chunks_indexed}")