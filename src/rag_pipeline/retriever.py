# src/rag_pipeline/retriever.py (FINAL SUBMISSION VERSION - Hybrid Search + KG + Re-ranking)

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import the shared OpenAI client for entity extraction
from .clients import openai_client

# Load environment variables (.env file) to get the Neo4j password
load_dotenv()

# --- Global variables for models and clients ---
print("Retriever: Loading all models and connecting to databases...")
EMBEDDING_MODEL = SentenceTransformer('./fine_tuned_model/')
CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("✅ Retriever: All models and clients loaded and ready.")

# --- Global variables for in-memory BM25 index ---
BM25_INDEX = None
CHUNK_CORPUS = {}

def get_db_connection():
    """Establishes connection to the PostgreSQL database for vector search."""
    try:
        return psycopg2.connect(host="localhost", port="5432", user="postgres", password="Sree@1997", dbname="rag_db")
    except psycopg2.OperationalError as e:
        print(f"FATAL: Could not connect to Postgres. Error: {e}")
        return None

def build_bm25_index():
    """Builds an in-memory BM25 (keyword) index from all the text chunks in Postgres."""
    global BM25_INDEX, CHUNK_CORPUS
    if BM25_INDEX is not None: return

    print("Building BM25 index from database content...")
    conn = get_db_connection()
    if not conn: return
    
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, content FROM chunks;")
    all_docs = cursor.fetchall()
    cursor.close()
    conn.close()

    if not all_docs: return

    CHUNK_CORPUS = {chunk_id: content for chunk_id, content in all_docs}
    tokenized_corpus = [content.split(" ") for chunk_id, content in all_docs]
    BM25_INDEX = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 index built with {len(all_docs)} documents.")

def extract_entities_from_question(question: str) -> list:
    """Uses an LLM to extract key entities (like company or people names) from the user's question."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert entity extractor. Your task is to extract the key proper nouns (people, companies, products) from the following question. If no entities are found, return an empty list. Output only a comma-separated list of the entities."},
                {"role": "user", "content": question}
            ]
        )
        content = response.choices[0].message.content
        # Handle cases where the LLM says "None" or similar
        if "none" in content.lower() or not content:
            return []
        entities = [e.strip() for e in content.split(",")]
        return [e for e in entities if e] # Remove any empty strings
    except Exception as e:
        print(f"  -> Entity extraction from question failed: {e}")
        return []

def query_knowledge_graph(entities: list) -> str:
    """Queries the Neo4j graph for facts and relationships connected to the extracted entities."""
    if not entities:
        return ""
    
    all_facts = []
    with NEO4J_DRIVER.session(database="neo4j") as session:
        for entity in entities:
            # This Cypher query finds all relationships where the given entity is the subject.
            query = """
            MATCH (e:Entity {name: $entity})-[r:RELATIONSHIP]->(o:Entity)
            RETURN e.name AS subject, r.type AS relation, o.name AS object
            """
            result = session.run(query, entity=entity)
            for record in result:
                # Format the graph triplet into a natural language sentence.
                fact = f"{record['subject']} {record['relation'].replace('_', ' ').lower()} {record['object']}."
                all_facts.append(fact)
    
    return " ".join(all_facts)

def retrieve_relevant_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Performs a four-stage HYBRID retrieval process:
    1. Dense Search (Vectors) + Sparse Search (Keywords)
    2. Knowledge Graph Search
    3. Combine all retrieved information
    4. Re-rank the combined context to find the best final results.
    """
    # --- Stage 1: DENSE & SPARSE RETRIEVAL ---
    conn = get_db_connection()
    if not conn: return []
    register_vector(conn)
    cursor = conn.cursor()
    
    # Dense results
    question_embedding = EMBEDDING_MODEL.encode(question)
    cursor.execute("SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20;", (question_embedding,))
    dense_results = {row[0] for row in cursor.fetchall()} # Use a set for automatic deduplication
    
    cursor.close()
    conn.close()

    # Sparse results
    tokenized_query = question.split(" ")
    doc_scores = BM25_INDEX.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:20]
    all_chunk_ids = list(CHUNK_CORPUS.keys())
    sparse_results = {CHUNK_CORPUS[all_chunk_ids[i]] for i in top_n_indices}

    # --- Stage 2: KNOWLEDGE GRAPH RETRIEVAL ---
    entities = extract_entities_from_question(question)
    kg_facts = query_knowledge_graph(entities)
    
    # --- Stage 3: COMBINE & RE-RANK ---
    # Combine all unique retrieved text chunks
    combined_context = list(dense_results.union(sparse_results))
    
    # Add the KG facts as a special, high-value piece of context
    if kg_facts:
        combined_context.append(kg_facts)

    if not combined_context:
        return []

    # The re-ranker scores the relevance of every piece of context against the original question.
    pairs = [[question, context] for context in combined_context]
    scores = CROSS_ENCODER_MODEL.predict(pairs, show_progress_bar=False)
    
    scored_results = sorted(list(zip(scores, combined_context)), key=lambda x: x[0], reverse=True)
    
    # Return the top_k results in the required dictionary format.
    top_chunks = scored_results[:top_k]
    # We add a special page_number "KG" if the fact came from the knowledge graph.
    return [{"content": content, "page_number": "Knowledge Graph" if content == kg_facts else "N/A"} for score, content in top_chunks]

# --- Initial Index Build ---
# This ensures that the BM25 index is built once when the application starts.
build_bm25_index()