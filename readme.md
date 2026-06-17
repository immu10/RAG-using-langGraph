# PDF querying Faiss

## Overview

An **agentic RAG** (Retrieval-Augmented Generation) system that lets you ask
natural-language questions about your own PDF documents. PDFs dropped into the
`docs/` folder are chunked, embedded locally, and stored in a persistent vector
database. At query time a [LangGraph](https://langchain-ai.github.io/langgraph/)
state machine first decides whether the question is relevant to the indexed
material, and only then retrieves the most similar chunks and asks an LLM
(`gpt-4o-mini`) to answer using that context. Irrelevant questions are rejected
early, so the model never hallucinates an answer from documents it doesn't have.

Key ideas:

- **Local embeddings** via `sentence-transformers/all-MiniLM-L6-v2` — no API cost for indexing.
- **Relevancy gate** — an LLM check short-circuits off-topic queries before retrieval.
- **Graph-based control flow** — each step (relevancy → retrieval → answer) is an explicit node.

## Architecture

```
                        +------------------------+
   PDFs in docs/ ---->  |   indexMaker()         |
                        |  load -> split (1200/  |
                        |  200) -> embed -> upsert|
                        +-----------+------------+
                                    |
                                    v
                          +-------------------+
                          |  Vector store     |
                          |  (persistent DB)  |
                          +---------+---------+
                                    ^
                                    | similarity_search
                                    |
   user question                    |
        |                            |
        v                            |
  +-----------+      No      +---------------+
  |  START    |---> relevancyCheck --------->| wrongAnswer |--> END
  +-----------+              |               +---------------+
                       Yes   |
                             v
                     +----------------+
                     | ChunkRetrieval |
                     +-------+--------+
                             |
                             v
                     +----------------+
                     |   chatbot      |  (gpt-4o-mini answers
                     |   (LLM answer) |   from retrieved chunks)
                     +-------+--------+
                             |
                             v
                            END
```

## Tech Stack

| Layer            | Technology                                            | Purpose                                       |
| ---------------- | ----------------------------------------------------- | --------------------------------------------- |
| Orchestration    | LangGraph                                             | State-machine control flow for the RAG agent  |
| Framework        | LangChain                                             | Loaders, splitters, vector-store integration  |
| LLM              | OpenAI `gpt-4o-mini` (via `langchain-openai`)         | Relevancy check + answer generation           |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)| Local document & query embeddings             |
| Vector store     | ChromaDB (persistent client)                          | Chunk storage and similarity search           |
| Document loading | LangChain `DirectoryLoader`                           | Reads PDFs from `docs/`                        |
| Config           | python-dotenv                                         | Loads `OPENAI_API_KEY` from `.env`            |
| Language         | Python 3                                              | —                                             |

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI key to a .env file
#    OPENAI_API_KEY=sk-...

# 3. Drop PDF files into a docs/ folder

# 4. Build the index and run a sample query
python agentic_rag.py
```

The first run calls `indexMaker()` to build the vector store, then `build_graph()`
compiles the LangGraph and runs an example query.

## Tags

`RAG` `LangGraph` `GPT` `LangChain` `ChromaDB` `embeddings` `vector-search` `PDF` `gpt-4o-mini` `HuggingFace`
