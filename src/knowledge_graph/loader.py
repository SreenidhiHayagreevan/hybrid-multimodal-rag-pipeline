# src/knowledge_graph/loader.py (FINAL VERSION - Neo4j)
import os
import sys
import json
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Load the Neo4j password from the .env file
load_dotenv()

# --- Neo4j Connection Details ---
# These must match your docker-compose.yml and .env files
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def load_data_into_neo4j(data_path="knowledge_graph_data.jsonl"):
    """
    Connects to the running Neo4j database, clears any old data, and loads
    the extracted (subject, relation, object) triplets.
    """
    
    # The driver is the main entry point to the database.
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        print(f"❌ Failed to create Neo4j driver. Is the database running? Error: {e}")
        return

    with driver.session(database="neo4j") as session:
        # 1. Clear the database for a fresh start to ensure no old data remains.
        print("Clearing existing knowledge graph data...")
        session.run("MATCH (n) DETACH DELETE n")

        # 2. Load the extracted triplets from the JSONL file.
        print(f"Loading data from '{data_path}'...")
        try:
            with open(data_path, "r") as f:
                triplets = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"❌ ERROR: Data file not found at '{data_path}'. Please run the extractor script first.")
            driver.close()
            return

        # 3. Insert the data into Neo4j using a Cypher query.
        for triplet in tqdm(triplets, desc="Loading Triplets into Neo4j"):
            subject = triplet.get("subject")
            # Cypher relations are typically uppercase and use underscores.
            relation = triplet.get("relation", "").upper().replace("-", "_")
            obj = triplet.get("object")
            
            if not all([subject, relation, obj]):
                continue

            # Cypher is the query language for Neo4j.
            # MERGE is a powerful command that means "find this pattern or create it."
            # It prevents creating duplicate nodes for the same entity.
            # We label all nodes as "Entity" and store the relationship type on the edge.
            query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:RELATIONSHIP {type: $relation}]->(o)
            """
            
            # We use parameters ($subject, etc.) to safely pass data to the query.
            session.run(query, subject=subject, object=obj, relation=relation)

    driver.close()
    print(f"\n✅ Successfully loaded {len(triplets)} triplets into the Neo4j knowledge graph.")

if __name__ == "__main__":
    if not NEO4J_PASSWORD:
        print("❌ ERROR: NEO4J_PASSWORD not found in .env file. Please set it and try again.")
    else:
        load_data_into_neo4j()