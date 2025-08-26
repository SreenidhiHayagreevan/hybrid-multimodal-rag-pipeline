# run_finetuning.py

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def run_finetuning():
    """
    Loads the generated dataset and fine-tunes the sentence-transformer model.
    """
    # 1. Load the pre-trained model you want to fine-tune
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading pre-trained model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2. Load your custom fine-tuning dataset
    dataset_path = "finetuning_dataset.jsonl"
    print(f"Loading dataset from: {dataset_path}")
    train_examples = []
    try:
        with open(dataset_path, "r") as f:
            for line in f:
                data = json.loads(line)
                train_examples.append(InputExample(texts=[data['query'], data['positive'], data['negative']]))
    except FileNotFoundError:
        print(f"ERROR: Dataset not found. Please run `create_finetune_dataset.py` first.")
        return
        
    print(f"Loaded {len(train_examples)} training examples.")

    # 3. Set up the data loader and loss function
    # The DataLoader will batch the examples for efficient training.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # MultipleNegativesRankingLoss is the standard and most effective loss function for this task.
    # It works on (anchor, positive, negative) triplets and pushes the anchor and positive
    # vectors closer while pushing the anchor and negative vectors apart.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Run the training
    num_epochs = 1
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up

    print(f"Starting fine-tuning for {num_epochs} epoch(s)...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path='./fine_tuned_model',
              show_progress_bar=True)

    print("\n✅ Fine-tuning complete.")
    print("The new, specialized model is saved in the './fine_tuned_model/' directory.")

if __name__ == "__main__":
    run_finetuning()