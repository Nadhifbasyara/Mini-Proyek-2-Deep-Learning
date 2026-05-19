import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# ======================
# SETUP
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "distilbert-base-uncased"
checkpoint_path = "./results/checkpoint-8654"

# ======================
# LOAD MODEL & TOKENIZER
# ======================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=7
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model = model.to(device)
model.eval()

# ======================
# LOAD TEST DATA
# ======================

print("Loading test data...")
test_df = pd.read_csv("test.csv")
print(f"Test data shape: {test_df.shape}")

# ======================
# INFERENCE
# ======================

predictions = []

print("Making predictions...")
with torch.no_grad():
    for idx, row in test_df.iterrows():
        essay_id = row["essay_id"]
        full_text = row["full_text"]
        
        # Tokenize
        inputs = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get prediction (argmax)
        pred_label = torch.argmax(logits, dim=1).item()
        # Convert 0-indexed to 1-indexed (labels are 1-7)
        pred_score = pred_label + 1
        
        predictions.append({
            "essay_id": essay_id,
            "score": pred_score
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(test_df)}")

# ======================
# SAVE RESULTS
# ======================

results_df = pd.DataFrame(predictions)
output_path = "submission_bonus_lora.csv"
results_df.to_csv(output_path, index=False)

print(f"\n✅ Predictions saved to {output_path}")
print(f"Total predictions: {len(results_df)}")
print("\nFirst 5 predictions:")
print(results_df.head())
