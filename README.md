# Naive RAG vs Advanced RAG

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain, Qdrant Cloud, and OpenAI's GPT-3.5.

## 🔍 Objective
To compare the performance of a simple (naive) RAG system that retrieves top-k chunks from a text corpus and sends them directly to a language model for answering questions.

## 📂 Structure

- `data/raw/`: contains `.txt` files about Moroccan culture.
- `ingest.py`: splits the documents, embeds them using `sentence-transformers`, and uploads to Qdrant.
- `query.py`: takes a user question, retrieves top 3 similar chunks, and uses OpenAI to answer.
- `results_naive.jsonl`: log of question-answer pairs.
- `.env`: stores secrets (not uploaded to GitHub).

## ⚙️ Setup

1. **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up `.env`**:
    ```env
    OPENAI_API_KEY=your-openai-key
    QDRANT_API_KEY=your-qdrant-key
    QDRANT_URL=https://your-cluster-url
    ```

3. **Ingest documents**:
    ```bash
    python ingest.py
    ```

4. **Ask a question**:
    ```bash
    python query.py
    ```

## 📌 Dependencies
- `openai`
- `qdrant-client`
- `langchain`
- `sentence-transformers`
- `python-dotenv`

## 🔐 Security
Secrets are handled using a `.env` file. Make sure to add it to `.gitignore`.

## 📥 Output
Answers are logged in `results_naive.jsonl`.

---

**Made by Mohammed ISMAILI**
