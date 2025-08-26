# src/rag_pipeline/orchestrator.py (FINAL SUBMISSION VERSION - High-Score Config)

import os
import sys
from typing import List, TypedDict

# Import Phoenix and the instrumentor for tracing
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

# Import LangGraph to build the agentic workflow
from langgraph.graph import StateGraph, END

# --- SETUP PHOENIX TRACING ---
# This block sets up the tracing. It is correctly implemented even if the UI
# has local display issues.
try:
    px.launch_app()
    LangChainInstrumentor().instrument()
    print("✅ Phoenix Tracing has been successfully launched and instrumented.")
except Exception as e:
    print(f"⚠️ Failed to launch Phoenix, proceeding without tracing. Error: {e}")
# -----------------------------

# Add the project's root directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import our custom-built, high-performance retriever and answer generator
from src.rag_pipeline.retriever import retrieve_relevant_chunks
from src.rag_pipeline.answer_generator import generate_answer

# --- Define the State of our Graph ---
# This is the "memory" of our application as it runs.
class GraphState(TypedDict):
    question: str
    documents: List[dict]
    generation: str

# --- Define the Nodes of our Graph ---
# This is the simple, high-performing two-node structure.

def retrieve_node(state: GraphState) -> GraphState:
    """
    The 'retrieve' node. It uses our most advanced retriever (fine-tuned + re-ranked)
    to get the best possible context for the question.
    """
    print("---NODE: Retrieving documents...---")
    question = state["question"]
    # This function now contains all the advanced logic (fine-tuning, re-ranking, etc.)
    documents = retrieve_relevant_chunks(question)
    return {"documents": documents}

def generate_node(state: GraphState) -> GraphState:
    """
    The 'generate' node. It takes the high-quality retrieved documents
    and generates the final answer.
    """
    print("---NODE: Generating answer...---")
    question = state["question"]
    documents = state["documents"]
    generation = generate_answer(question, documents)
    return {"generation": generation}

# --- Build and Compile the Graph ---
workflow = StateGraph(GraphState)

# Add the two core nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define the simple, linear flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
print("✅ LangGraph workflow compiled successfully.")

# --- Main Execution Block ---
if __name__ == "__main__":
    test_question = "What was the revenue for Intelligent Cloud?"
    graph_input = {"question": test_question}

    print(f"\n🚀 --- Running RAG pipeline for question: '{test_question}' ---")
    
    # Use .invoke() for a single, robust run of the entire graph.
    final_state = app.invoke(graph_input)
    
    print("\n✅ --- FINAL GENERATED ANSWER ---")
    if final_state:
        print(final_state.get("generation"))
    
    if session := px.active_session():
        print("\n📈 --- PHOENIX TRACING ---")
        print("Phoenix is running. Open the URL below to view the trace:")
        print(f"Phoenix UI URL: {session.url}")
        print("\nPress Ctrl+C in the terminal to stop this script and the Phoenix server.")
        
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nScript finished. Shutting down Phoenix server.")
    else:
        print("\n⚠️ Phoenix was not launched. Skipping trace view.")