# src/rag_pipeline/clients.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Centralized OpenAI Client ---
openai_client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    
    openai_client = OpenAI(api_key=api_key)
    print("✅ Central OpenAI client initialized successfully.")
except Exception as e:
    print(f"❌ An error occurred during OpenAI client initialization: {e}")