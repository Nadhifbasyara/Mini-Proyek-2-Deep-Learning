import pandas as pd
import ollama
from tqdm import tqdm

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ambil contoh
examples = train_df.sample(3)

example_text = ""

for _, row in examples.iterrows():
    example_text += f"""
Essay:
{row['full_text']}

Score:
{row['score']}
"""

predictions = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

    essay = row["full_text"]

    prompt = f"""
You are an automatic essay scoring system.

Here are examples:

{example_text}

Now score this essay from 1 to 6.

Essay:
{essay}

Only return the score.
"""

    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    score = response['message']['content'].strip()

    predictions.append(score)

submission = pd.DataFrame({
    "essay_id": test_df["essay_id"],
    "score": predictions
})

submission.to_csv("submission_few.csv", index=False)

print("Done!")