import pandas as pd
import ollama
from tqdm import tqdm

# load data
test_df = pd.read_csv("test.csv")

predictions = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

    essay = row["full_text"]

    prompt = f"""
    You are an essay scoring system.

    Score this essay from 1 to 6.

    Essay:
    {essay}

    Only return the score number.
    """

    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    score = response['message']['content'].strip()

    predictions.append(score)

# save
submission = pd.DataFrame({
    "essay_id": test_df["essay_id"],
    "score": predictions
})

submission.to_csv("submission_zero.csv", index=False)

print("Done!")