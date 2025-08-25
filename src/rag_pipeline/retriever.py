# src/rag_pipeline/retriever.py (FINAL UPGRADED VERSION - with Re-ranking)

import os
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
# Import both SentenceTransformer for the initial search and CrossEncoder for re-ranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Global variables ---
# Load all models once when the script is first imported. This is a crucial
# optimization to prevent reloading these large models on every single question.
print("Retriever: Loading embedding model (for fast initial search)...")
# This model creates the vectors for our initial database search.
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

print("Retriever: Loading Cross-Encoder model (for accurate re-ranking)...")
# This is a more powerful model trained specifically to score the relevance
# of a question-document pair. It's slower but much more accurate.
CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("✅ Retriever: All models loaded and ready.")

def get_db_connection():
    """Establishes and returns a direct connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="localhost", port="5432", user="postgres",
            password="Sree@1997", dbname="rag_db" # IMPORTANT: Use your password
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"FATAL: Could not connect to the database. Error: {e}")
        return None

def retrieve_relevant_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Performs a two-stage retrieval process to find the most relevant text chunks:

    1.  **FETCH (Candidate Selection):** A fast vector search retrieves a broad set
        of potentially relevant documents (e.g., the top 20).
    2.  **RE-RANK (Refinement):** A more powerful Cross-Encoder model scores the relevance
        of each candidate document against the question and re-ranks them to find the
        true best matches.
    """
    if not question:
        return []

    conn = get_db_connection()
    if not conn:
        return []
    
    register_vector(conn)
    cursor = conn.cursor()

    try:
        # --- Stage 1: FETCH ---
        # We retrieve a larger number of candidates than we need for the final answer.
        candidate_count = 20
        question_embedding = EMBEDDING_MODEL.encode(question)
        
        cursor.execute(
            "SELECT content, page_number, chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s;",
            (question_embedding, candidate_count)
        )
        candidate_results = cursor.fetchall()
        
        if not candidate_results:
            return []

        # --- Stage 2: RE-RANK ---
        # Create pairs of [question, chunk_content] for the Cross-Encoder.
        pairs = [[question, row[0]] for row in candidate_results]
        
        # The Cross-Encoder predicts a highly accurate relevance score for each pair.
        scores = CROSS_ENCODER_MODEL.predict(pairs, show_progress_bar=False)
        
        # Combine the original database results with their new, more accurate scores.
        scored_results = list(zip(scores, candidate_results))
        
        # Sort the results by the new score in descending order (higher score is better).
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top_k best results from the re-ranked list.
        top_results = scored_results[:top_k]

        # Format the final list of chunks to be passed to the LLM.
        retrieved_chunks = [{
            "content": row[1][0],
            "page_number": row[1][1],
            "chunk_id": row[1][2]
        } for row in top_results]
        
        return retrieved_chunks

    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return []
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# Main block for direct testing of this script's functionality.
if __name__ == "__main__":
    print("\n--- Testing the Upgraded Retriever (with Re-ranking) ---")
    test_question = "What was the revenue for Intelligent Cloud?"
    print(f"Test Question: {test_question}")
    
    relevant_chunks = retrieve_relevant_chunks(test_question)
    
    if relevant_chunks:
        print(f"\nFound {len(relevant_chunks)} re-ranked chunks:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i+1} (from Page {chunk['page_number']}) ---")
            # Print a snippet of the content to keep the output clean
            print(chunk['content'][:250] + "...")
    else:
        print("No relevant chunks were found.")