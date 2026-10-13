# Tiny RAG Agent

A small agentic-RAG assistant that runs in one Python file. It indexes a folder of notes and lets Claude search them before answering, with citations.

```
you> why does my model do great on training data but badly on new data?
  [search] {"query": "model performs well on training data but poorly on new data"}
  [search] {"query": "how to reduce overfitting"}

That pattern is overfitting [overfitting.md#0]. Common fixes are more data, a simpler
model, and regularization [regularization.md#1] ...
```

## How it works

```
docs/*.md ──> chunk (80 words, 20 overlap) ──> TF-IDF index ──> search_docs tool ──> Claude ──> cited answer
```

1. **Ingest:** split every `.md` or `.txt` file into overlapping word windows.
2. **Index:** build a TF-IDF matrix with scikit-learn. There's no embedding model, no vector database and no downloads.
3. **Retrieve:** `search_docs` ranks chunks by cosine similarity.
4. **Generate:** Claude decides when to search, can rewrite its query and search again, and answers only from what it found.

Because retrieval is a tool and not a fixed step, the agent can run several searches for a multi-part question or say when the docs don't cover it.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python agent.py            # uses the sample ML notes in ./docs
python agent.py my_notes/  # or point it at your own folder of .md/.txt files
```

## Design notes

- Dependencies are `anthropic` and `scikit-learn`, which install in tens of MB. Swapping TF-IDF for embeddings later only means changing `search_docs`.
- Every passage carries a `[file#chunk]` ID, so answers can be traced back to their source.
- The system prompt tells the model to answer only from the documents and to admit gaps.
- A step cap stops runaway search loops.
