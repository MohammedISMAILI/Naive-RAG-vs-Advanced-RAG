from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import uuid

# Connect to Qdrant Cloud
client = QdrantClient(
    url="https://3a8f955b-bf73-4a01-a662-f601df2b60b1.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="qdrant_...",
)

# Load text files
docs = []
for fname in os.listdir("data/raw/"):
    with open(f"data/raw/{fname}", "r", encoding="utf-8") as f:
        docs.append(Document(page_content=f.read()))

# Chunking
splitter = CharacterTextSplitter(chunk_size=512, separator="\n")
chunks = splitter.split_documents(docs)

# Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode([doc.page_content for doc in chunks])

# Recreate the collection
collection_name = "naive_corpus"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# Upload vectors and payloads
payloads = [{"text": doc.page_content} for doc in chunks]
ids = [str(uuid.uuid4()) for _ in chunks]

client.upload_collection(
    collection_name=collection_name,
    vectors=vectors,
    payload=payloads,
    ids=ids
)

print("✅ Uploaded", len(chunks), "documents to Qdrant Cloud.")
