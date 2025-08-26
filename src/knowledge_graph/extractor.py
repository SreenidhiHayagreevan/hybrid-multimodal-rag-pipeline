# src/knowledge_graph/extractor.py

import os
import sys
import json
from tqdm import tqdm

# Add the project root to the path so we can import our other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the shared OpenAI client and the database connection function
from src.rag_pipeline.answer_generator import client as openai_client
from src.indexing.indexer import get_db_connection

def extract_entities_and_relations(chunk_content: str) -> list:
    """Uses an LLM to extract structured (subject, relation, object) triplets from a text chunk."""
    if not openai_client:
        print("OpenAI client not initialized. Cannot extract KG data.")
        return []
    
    try:
        # This prompt is specifically engineered to ask the LLM for structured JSON output.
        # This is a powerful technique called "Function Calling" or "Tool Use".
        prompt = f"""
        You are an expert data extractor for a Knowledge Graph. Your task is to identify key entities and their relationships from the following text passage.
        Extract the information as a list of JSON objects. Each object must have three keys: "subject", "relation", and "object".
        
        Focus on important financial and organizational data.
        - Valid entity types: 'Company', 'Product', 'Revenue', 'Date', 'Person'.
        - Valid relations: 'HAS_REVENUE', 'ANNOUNCED_ON', 'IS_CEO_OF', 'PRODUCES'.
        
        Example Output Format:
        {{
            "triplets": [
                {{"subject": "Microsoft", "relation": "HAS_REVENUE", "object": "$24.3 billion"}},
                {{"subject": "Satya Nadella", "relation": "IS_CEO_OF", "object": "Microsoft"}}
            ]
        }}

        --- TEXT PASSAGE ---
        {chunk_content}
        --- END TEXT PASSAGE ---

        Please provide the extracted information in the specified JSON format.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"}, # Use the OpenAI API's JSON mode
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the JSON string returned by the API
        result_json = json.loads(response.choices[0].message.content)
        
        # The prompt asks for a 'triplets' key, so we extract that list.
        return result_json.get("triplets", [])

    except Exception as e:
        print(f"  -> An error occurred during KG data extraction: {e}")
        return []

def run_extraction():
    """Fetches a sample of chunks from the DB, extracts KG data, and saves it to a file."""
    print("Connecting to the database to fetch chunks for KG extraction...")
    conn = get_db_connection()
    if not conn: return

    cursor = conn.cursor()
    # To manage API costs and time, we will only build the KG from a sample of 200 chunks.
    # For a full production system, you would process all chunks.
    cursor.execute("SELECT content FROM chunks LIMIT 200;")
    sample_chunks = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    if not sample_chunks:
        print("No chunks found in the database.")
        return

    print(f"Found {len(sample_chunks)} chunks to process. Extracting entities and relations...")
    
    output_file = "knowledge_graph_data.jsonl"
    all_triplets = []

    # Use a .jsonl (JSON Lines) format, which is robust for streaming data.
    with open(output_file, "w") as f:
        for chunk in tqdm(sample_chunks, desc="Extracting KG Triplets"):
            triplets = extract_entities_and_relations(chunk)
            if triplets and isinstance(triplets, list):
                all_triplets.extend(triplets)
                for triplet in triplets:
                    # Write each JSON object as a new line in the file.
                    f.write(json.dumps(triplet) + "\n")

    print(f"\n✅ Successfully extracted {len(all_triplets)} triplets.")
    print(f"Knowledge graph data saved to: {output_file}")

if __name__ == "__main__":
    run_extraction()