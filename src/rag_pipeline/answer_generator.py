# src/rag_pipeline/answer_generator.py (FINAL CORRECTED VERSION)

import os
import sys

# --- FIX: Import the shared client from our new central clients.py file ---
from .clients import openai_client

# We no longer need the retriever import here, which breaks the circular dependency.
# sys.path.append(...) is also no longer needed for this file's logic.

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generates a final answer using the shared OpenAI client, based on the
    user's question and the retrieved context.
    """
    if not openai_client:
        return "OpenAI client could not be initialized. Please check clients.py and your .env file."

    if not context_chunks:
        return "I could not find any relevant information in the documents to answer your question."

    # Combine the content of all retrieved chunks into a single string for the prompt
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

    try:
        response = openai_client.chat.completions.create(
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


# The __main__ block is removed from this file as it is now intended only as a module.
# The main entry points for running the pipeline are orchestrator.py and run_evaluation.py.