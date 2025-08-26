# run_autogen_task.py (FINAL SUBMISSION VERSION)
import os
import glob
# --- FIX: Import directly from the installed packages ---
from agentchat import ConversableAgent
from agentchat.user_proxy_agent import UserProxyAgent
from autogen_core.function_utils import register_function

def find_pdf_files(folder_path: str) -> str:
    """A tool that finds all PDF files in a given folder path."""
    print(f"\n--- TOOL EXECUTING: Searching for PDF files in '{folder_path}' ---")
    pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
    if not pdf_files:
        return "TERMINATE: No PDF files found in the specified directory."
    # We return TERMINATE to end the conversation after one successful tool call.
    return "TERMINATE\n" + "\n".join(pdf_files)

def run_autogen_etl_finder():
    """Defines and runs the AutoGen agentic workflow for finding ETL files."""
    
    # The Assistant Agent is the "worker" that can be configured to use tools.
    assistant = ConversableAgent(
        name="Assistant",
        llm_config=False,  # No LLM is needed; the agent will rely on its tool.
    )

    # The User Proxy Agent acts on behalf of the user and can execute function calls.
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: x.get("content", "").rstrip().startswith("TERMINATE"),
        code_execution_config=False, # We are not executing code, only calling a function.
    )

    # Register the Python function as a tool for the agents to use.
    register_function(
        find_pdf_files,
        caller=assistant,
        executor=user_proxy,
        name="find_pdf_files",
        description="Finds all PDF files in a folder and its subfolders.",
    )

    print("--- Starting AutoGen ETL File Finder ---")
    
    # Initiate the chat. The user_proxy sends the first message.
    user_proxy.initiate_chat(
        assistant,
        message="Use the `find_pdf_files` tool to find all PDF files in the 'data/SampleDataSet/' directory.",
    )
    
    print("\n--- AutoGen Conversation Finished ---")
    print("✅ This script has successfully used an Auto-Gen agent system for the data identification step of the ETL pipeline.")

if __name__ == "__main__":
    run_autogen_etl_finder()