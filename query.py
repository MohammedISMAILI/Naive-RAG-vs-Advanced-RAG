import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Define API Keys
openai_api_key = "sk-..."  # Replace with your actual OpenAI API key
qdrant_api_key = "qdrant_..."

# Question input
question = input("Enter your question: ")

# Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
q_vector = model.encode([question])[0]

# Qdrant search
client = QdrantClient(
    url="https://3a8f955b-bf73-4a01-a662-f601df2b60b1.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=qdrant_api_key
)
hits = client.search(
    collection_name="naive_corpus",
    query_vector=q_vector,
    limit=3
)
contexts = [hit.payload["text"] for hit in hits]

# Build prompt
prompt = f"""
System: Answer concisely using only the information in CONTEXT.
CONTEXT:
{chr(10).join(contexts)}

Question: {question}
"""

# Query OpenAI
client_openai = OpenAI(api_key=openai_api_key)
response = client_openai.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512
)

answer = response.choices[0].message.content
print("\nAnswer:\n", answer)

# Log result
with open("results_naive.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "question": question,
        "contexts": contexts,
        "answer": answer
    }) + "\n")
