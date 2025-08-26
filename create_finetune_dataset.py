# create_finetune_dataset.py

import os
import sys
import json
import random
from tqdm import tqdm

# Add src to path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.rag_pipeline.answer_generator import client as openai_client
from src.indexing.indexer import get_db_connection

def generate_question_for_chunk(chunk_content: str) -> str:
    """Uses an LLM to generate a realistic question for a given text chunk."""
    if not openai_client:
        return None
    
    try:
        prompt = f"""
        You are a helpful assistant that creates training data for a RAG system.
        Based on the following text passage from a document, please generate one concise, high-quality question that this passage could directly answer.
        The question should be something a user trying to understand the document would realistically ask.
        Do not ask questions that cannot be answered by the text. Output only the question, with no preamble.

        --- TEXT PASSAGE ---
        {chunk_content}
        --- END TEXT PASSAGE ---

        QUESTION:
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # GPT-4 is better but more expensive
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  -> Error generating question: {e}")
        return None

def create_dataset():
    """
    Connects to the database, fetches all chunks, and generates a
    (query, positive_passage, negative_passage) dataset for fine-tuning.
    """
    print("Connecting to database to fetch all indexed chunks...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM chunks;")
    all_chunks = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not all_chunks:
        print("No chunks found in the database. Please run the indexer first.")
        return

    print(f"Found {len(all_chunks)} chunks. Generating training triplets...")
    
    # We will save the data in a .jsonl file, one JSON object per line.
    output_file = "finetuning_dataset.jsonl"
    
    with open(output_file, "w") as f:
        for i in tqdm(range(len(all_chunks)), desc="Generating Triplets"):
            positive_passage = all_chunks[i]

            # Generate a query for our positive passage
            query = generate_question_for_chunk(positive_passage)
            if not query:
                continue

            # Find a negative passage. A "hard" negative is one from the same
            # context that is NOT the positive one. We will just pick a random one.
            negative_passage_index = random.randint(0, len(all_chunks) - 1)
            # Ensure the negative is not the same as the positive
            while negative_passage_index == i:
                negative_passage_index = random.randint(0, len(all_chunks) - 1)
            negative_passage = all_chunks[negative_passage_index]

            # Write the triplet to the file
            f.write(json.dumps({
                "query": query,
                "positive": positive_passage,
                "negative": negative_passage
            }) + "\n")

    print(f"\n✅ Successfully created fine-tuning dataset at: {output_file}")
    print("This dataset contains (query, positive, negative) triplets.")

if __name__ == "__main__":
    create_dataset()