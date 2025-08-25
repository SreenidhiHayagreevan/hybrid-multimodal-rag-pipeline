# src/indexing/setup_db.py (Direct Connection Version)
import psycopg2

def setup_database():
    """Sets up the database schema for the RAG pipeline."""
    conn = None
    try:
        # Connect directly to your PostgreSQL database running in Docker
        conn = psycopg2.connect(
            host="localhost",      # Use TCP/IP host
            port="5432",           # Use TCP/IP port
            user="postgres",       # Your username
            password="Sree@1997",  # Your password
            dbname="rag_db"        # Your database name
        )
        cursor = conn.cursor()
        print("Successfully connected to the database.")

        # Install the pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Vector extension is enabled.")

        # Create the table to store document chunks
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            page_number INTEGER NOT NULL,
            chunk_id VARCHAR(255) UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384)
        );
        """)
        print("Table 'chunks' has been created successfully.")

        # Commit the changes
        conn.commit()
        cursor.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    setup_database()