import json


def save_jsonl(df, output_path):
    # Converts a DataFrame with 'sentence' and 'label' columns into JSONL format for LLM fine-tuning.
    system_msg = "You are a financial sentiment classifier. Classify the sentence as Positive, Neutral, or Negative."

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": row["sentence"]},
                    {"role": "assistant", "content": row["label"]},
                ]
            }
            f.write(json.dumps(example) + "\n")
