from ragas import evaluate, answer_relevancy, context_precision, faithfulness
import json
import pandas as pd

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]

def format_for_ragas(results):
    return [
        {
            "question": r["question"],
            "contexts": r["contexts"],
            "answer": r["answer"],
            "ground_truth": [[]]
        } for r in results
    ]

naive = load_jsonl("results_naive.jsonl")
adv = load_jsonl("results_advanced.jsonl")

naive_dataset = format_for_ragas(naive)
adv_dataset = format_for_ragas(adv)

metrics = [answer_relevancy, context_precision, faithfulness]
naive_scores = evaluate(naive_dataset, metrics=metrics)
advanced_scores = evaluate(adv_dataset, metrics=metrics)

# Save
df = pd.DataFrame([naive_scores, advanced_scores], index=["naive", "advanced"])
df.to_csv("evaluation.csv")
print(df)
