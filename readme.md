
# 🧠 Agentic RAG — Automated Retrieval-Augmented Generation using LangGraph

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline using **LangGraph**, **LangChain**, **ChromaDB**, and **OpenAI APIs**.
It automates document retrieval, relevance checking, and response generation through a dynamic graph of reasoning nodes.

---

## 🚀 Overview

The system works as an **intelligent document question-answering agent**.
It indexes local PDF documents, determines whether a query is relevant to the dataset, retrieves matching text chunks from a vector database, and generates an LLM-based answer.

The workflow is built using a **graph-based architecture** (via `StateGraph` from `langgraph`), with nodes representing distinct steps in the RAG pipeline.

---

## ⚙️ Workflow Summary

### 1. **Index Creation**

* The function `indexMaker()` loads all PDFs from the `docs/` directory.
* Each document is split into smaller text chunks using `RecursiveCharacterTextSplitter`.
* Embeddings are generated via the **HuggingFace model** `sentence-transformers/all-MiniLM-L6-v2`.
* The chunks are stored in a persistent **ChromaDB** collection.

### 2. **Graph Nodes**

The graph includes the following key nodes:

| Node             | Function            | Description                                                        |
| ---------------- | ------------------- | ------------------------------------------------------------------ |
| `relevancyCheck` | LLM-based filter    | Determines if a user query relates to available documents.         |
| `ChunkRetrieval` | Vector store search | Retrieves the most relevant text chunks using semantic similarity. |
| `chatbot`        | LLM response        | Generates a contextual answer based on retrieved chunks.           |
| `wrongAnswer`    | Fallback            | Returns a default message for irrelevant queries.                  |

### 3. **Graph Flow**

```
START
  ↓
relevancyCheck ──┬── True → ChunkRetrieval → chatbot → END
                  └── False → wrongAnswer → END
```

---

## 🧩 Key Components

### 🗃️ Chroma Vector Store

Stores and retrieves text embeddings for document chunks.
All data persists in the local directory `chroma_db/`.

### 🧠 Models Used

* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM:** `gpt-4o-mini` (via `ChatOpenAI`)

### 🧱 Frameworks & Libraries

* `langchain`
* `langgraph`
* `chromadb`
* `langchain_openai`
* `langchain_huggingface`
* `dotenv`

---

## 🧰 How to Run

### 1. **Install Dependencies**

```bash
pip install langchain langgraph langchain-openai langchain-huggingface chromadb python-dotenv
```

### 2. **Set up Environment Variables**

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. **Add Documents**

Place all `.pdf` files inside a folder named `docs/`.

### 4. **Run the Script**

```bash
python agentic_rag.py
```

This will:

* Create a vector index (`indexMaker`)
* Build and run the graph (`build_graph`)
* Print the LLM’s answer to a sample query

---

## 🧠 Example Output

```bash
how much money do you start with in monopoly?

> The starting amount in Monopoly is $1500, distributed as follows: two $500s, two $100s, two $50s, six $20s, five $10s, five $5s, and five $1s.
```

---

## 🔮 Future Improvements ideas if anyone wants to contribute

* Integrate **multi-format loaders** (e.g., `.txt`, `.docx`).
* Add **dynamic document categories** for relevance checks.
* Implement **feedback loops** for self-correcting retrieval.
* Add **frontend interface** (e.g., Streamlit or FastAPI).

---

## 🧑‍💻 Author

Developed as a research prototype demonstrating **Agentic RAG pipelines** using **LangGraph** and **ChromaDB**.

---
