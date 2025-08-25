# src/rag_pipeline/answer_generator.py (FINAL VERSION - Direct Initialization)
import os
import sys
from openai import OpenAI

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import our retriever function
from src.rag_pipeline.retriever import retrieve_relevant_chunks

# --- Direct Initialization of the OpenAI client ---
# This is a workaround for a local environment conflict that prevents .env loading.
API_KEY = "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"

client = None
try:
    if not API_KEY or "sk-proj" not in API_KEY:
         print("ERROR: The API_KEY is missing or invalid. Please paste your key directly into the script.")
    else:
        client = OpenAI(api_key=API_KEY)
        print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"An error occurred during OpenAI client initialization: {e}")


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generates a final answer using an LLM, based on the user's question
    and the retrieved context.
    """
    if not client:
        return "OpenAI client could not be initialized. Please check the API_KEY variable in the script."

    if not context_chunks:
        return "I could not find any relevant information to answer your question."

    # --- Create the Prompt ---
    context_str = "\n\n".join([chunk['content'] for chunk in context_chunks])

    prompt = f"""
    You are a helpful assistant. Answer the user's question based *only* on the provided context.
    Do not use any external knowledge.
    If the answer is not available in the context, say "I could not find the answer in the provided documents."
    Cite the sources used to answer the question from the provided metadata.

    CONTEXT:
    ---
    {context_str}
    ---

    QUESTION: {question}

    ANSWER and SOURCES:
    """

    # --- Call the LLM ---
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred while communicating with the OpenAI API: {e}"


# This block allows us to run the full RAG pipeline from this file
if __name__ == "__main__":
    test_question = "What was the revenue for Intelligent Cloud?"
    
    print(f"--- Running Full RAG Pipeline for Question: '{test_question}' ---")

    # 1. Retrieve relevant context from the database
    print("\nStep 1: Retrieving relevant documents...")
    relevant_chunks = retrieve_relevant_chunks(test_question)
    
    if not relevant_chunks:
        print("Pipeline finished: No relevant documents found.")
    else:
        print(f"Found {len(relevant_chunks)} relevant chunks.")

        # 2. Generate the final answer using the LLM
        print("\nStep 2: Generating final answer with LLM...")
        final_answer = generate_answer(test_question, relevant_chunks)
        
        print("\n--- FINAL ANSWER ---")
        print(final_answer)