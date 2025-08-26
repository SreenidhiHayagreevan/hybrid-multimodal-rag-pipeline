# src/evaluation/run_evaluation.py (FINAL SUBMISSION VERSION - Corrected Imports)

import os
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the necessary components from your RAG pipeline
from src.rag_pipeline.orchestrator import app
# --- FIX: Import the shared client from our new central clients.py file ---
from src.rag_pipeline.clients import openai_client

def llm_as_judge(question: str, ground_truth: str, predicted_answer: str) -> bool:
    """
    Uses a Large Language Model as a judge to evaluate if the predicted answer is correct.
    """
    if not openai_client:
        print("WARNING: OpenAI client not available for judging. Skipping check.")
        return False
    
    try:
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
            temperature=0,
            max_tokens=5
        )
        
        judge_response = response.choices[0].message.content.strip().lower()
        return "yes" in judge_response

    except Exception as e:
        print(f"\nAn error occurred during the LLM-as-a-judge call: {e}")
        return False


def run_evaluation():
    """
    Runs the full evaluation, calculates metrics, and saves the detailed
    results to a CSV file for analysis.
    """
    # --- 1. Load the Evaluation Dataset ---
    eval_csv_path = "data/SampleDataSet/SampleQuestions/questions_with_partial_answers.csv"
    try:
        eval_df = pd.read_csv(eval_csv_path, encoding='latin1')
        eval_df.dropna(subset=['Answer'], inplace=True)
        print(f"✅ Loaded {len(eval_df)} questions with ground truth answers for evaluation.")
    except Exception as e:
        print(f"❌ Error loading or parsing the evaluation file: {e}")
        return

    # Create a list to store detailed results for saving later
    results_list = []
    
    predictions = []
    ground_truths = []
    correct_count = 0

    print("🚀 Starting evaluation with LLM-as-a-Judge. This will take several minutes...")
    
    # --- 2. Iterate Through Questions and Evaluate ---
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="LLM Judge Evaluation"):
        question = row['Question']
        ground_truth = str(row['Answer'])

        graph_input = {"question": question}
        final_state = app.invoke(graph_input)
        predicted_answer = final_state.get("generation", "No answer generated.")

        is_correct = llm_as_judge(question, ground_truth, predicted_answer)

        if is_correct:
            predictions.append(1)
            correct_count += 1
        else:
            predictions.append(0)
        
        ground_truths.append(1)
        
        # Append the detailed results for this question to our list
        results_list.append({
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })
        
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

    # --- 4. Save the detailed results to a CSV file ---
    print("\n💾 Saving detailed evaluation results to CSV...")
    try:
        results_df = pd.DataFrame(results_list)
        output_csv_path = "evaluation_results.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"✅ Results saved to '{output_csv_path}'")
    except Exception as e:
        print(f"❌ Failed to save results to CSV. Error: {e}")

if __name__ == "__main__":
    run_evaluation()