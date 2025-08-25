# src/evaluation/run_evaluation.py (FINAL SUBMISSION VERSION)

import os
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Add the project's root directory to the Python path
# This ensures we can import our other custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the compiled LangGraph app from the orchestrator
from src.rag_pipeline.orchestrator import app
# Import the initialized OpenAI client from the answer_generator for our judge
# This reuses the client that has the API key correctly configured
from src.rag_pipeline.answer_generator import client as openai_client

def llm_as_judge(question: str, ground_truth: str, predicted_answer: str) -> bool:
    """
    Uses a Large Language Model as a judge to evaluate if the predicted answer is correct.

    This is a powerful technique for evaluating generative models, as it can understand
    paraphrasing and semantic similarity, unlike simple string matching.

    Returns:
        True if the LLM judges the answer as correct, False otherwise.
    """
    if not openai_client:
        print("WARNING: OpenAI client not available for judging. Skipping check.")
        return False
    
    try:
        # The prompt that instructs the LLM on how to act as a judge.
        prompt = f"""
        You are an impartial judge. Your task is to evaluate if a "Predicted Answer" is a correct and factual response to a "Question", based on a "Ground Truth Answer".

        - The Predicted Answer does not need to be a word-for-word copy of the Ground Truth.
        - It must contain the same core facts and information.
        - Paraphrasing or summarizing the Ground Truth is acceptable and should be considered correct.
        - Minor differences in formatting (e.g., "$24.3 billion" vs "$24,300 million") are acceptable if the values are equivalent.

        Question: "{question}"
        Ground Truth Answer: "{ground_truth}"
        Predicted Answer: "{predicted_answer}"

        Based on these rules, is the Predicted Answer correct? Answer with only the single word 'yes' or 'no'.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, # Use a low temperature for deterministic and consistent judging
            max_tokens=5   # Limit the response to a single word for easy parsing
        )
        
        judge_response = response.choices[0].message.content.strip().lower()
        # Return True if the judge's response contains "yes"
        return "yes" in judge_response

    except Exception as e:
        print(f"\nAn error occurred during the LLM-as-a-judge call: {e}")
        return False


def run_evaluation():
    """
    Runs the full evaluation using an LLM as the judge for correctness.
    """
    # --- 1. Load the Evaluation Dataset ---
    eval_csv_path = "data/SampleDataSet/SampleQuestions/questions_with_partial_answers.csv"
    try:
        # Use encoding='latin1' to handle special characters and the correct column names.
        eval_df = pd.read_csv(eval_csv_path, encoding='latin1')
        eval_df.dropna(subset=['Answer'], inplace=True)
        print(f"✅ Loaded {len(eval_df)} questions with ground truth answers for evaluation.")
    except Exception as e:
        print(f"❌ Error loading or parsing the evaluation file: {e}")
        return

    predictions = []
    ground_truths = []
    correct_count = 0

    print("🚀 Starting evaluation with LLM-as-a-Judge. This will take several minutes...")
    
    # --- 2. Iterate Through Questions and Evaluate ---
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="LLM Judge Evaluation"):
        question = row['Question']
        ground_truth = str(row['Answer'])

        # Run your RAG pipeline to get the predicted answer
        graph_input = {"question": question}
        final_state = app.invoke(graph_input)
        predicted_answer = final_state.get("generation", "No answer generated.")

        # Use the LLM to judge the correctness of the answer
        if llm_as_judge(question, ground_truth, predicted_answer):
            predictions.append(1) # Correct
            correct_count += 1
        else:
            predictions.append(0) # Incorrect
        
        ground_truths.append(1)
        
        # Display a running accuracy score in the progress bar's description
        tqdm.write(f"Running Accuracy: {correct_count / len(ground_truths):.2%}")

    # --- 3. Calculate Final Metrics ---
    if not ground_truths:
        print("⚠️ No questions were evaluated.")
        return
        
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, zero_division=0)
    recall = recall_score(ground_truths, predictions, zero_division=0)
    f1 = f1_score(ground_truths, predictions, zero_division=0)

    print("\n" + "="*50)
    print("--- ✅ FINAL EVALUATION METRICS (LLM-as-a-Judge) ---")
    print(f"Total Questions Evaluated: {len(ground_truths)}")
    print(f"Accuracy (Correct Answers): {accuracy:.2%}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()