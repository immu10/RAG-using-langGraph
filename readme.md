# Agentic RAG using LangGraph

A small Retrieval-Augmented Generation (RAG) agent built with [LangGraph](https://langchain-ai.github.io/langgraph/). It indexes a folder of PDFs into a local [Chroma](https://www.trychroma.com/) vector store, then answers questions about them — but first checks whether the question is even relevant to the indexed documents and short-circuits if it isn't.

## How it works

The agent is a LangGraph state machine with the following flow:

```
START
  │
  ▼
relevancyCheck ──(not relevant)──► wrongAnswer ──► END
  │
  (relevant)
  ▼
ChunkRetrieval ──► chatbot ──► END
```

- **relevancyCheck** — asks an LLM whether the question relates to the known documents (e.g. *monopoly*, *chess*).
- **ChunkRetrieval** — runs a similarity search against the Chroma vector store.
- **chatbot** — answers the question using the retrieved chunks as context.
- **wrongAnswer** — returns a canned "not relevant" reply when the question is off-topic.

Embeddings are generated locally with `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace), and answers are generated with OpenAI's `gpt-4o-mini`.

## Setup

1. **Clone and enter the repo**

   ```bash
   git clone <repo-url>
   cd RAG-using-langGraph
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows (PowerShell)
   # source .venv/bin/activate # macOS / Linux
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key** in a `.env` file at the project root:

   ```
   OPENAI_API_KEY=sk-...
   ```

4. **Add source documents** — place the PDFs you want to query in a `docs/` folder:

   ```
   docs/
     monopoly.pdf
     chess.pdf
   ```

## Usage

Run the script. It will (re)build the index from `docs/` and then run a sample query:

```bash
python agentic_rag.py
```

The vector store is persisted to a local `chroma_db/` directory, so subsequent runs reuse the index. To change the question, edit the `agentic_rag.invoke(...)` call in `build_graph()`.

## Project structure

```
agentic_rag.py     # indexing + LangGraph agent
requirements.txt   # Python dependencies
.gitignore
```

## Notes

- `indexMaker()` upserts chunks keyed by a SHA-512 hash of their content, so re-running won't create duplicates.
- The list of known document topics in `relevancyCheck` is currently hard-coded (`["monopoly", "chess"]`) — update it to match your own documents.
