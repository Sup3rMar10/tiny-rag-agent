"""A tiny agentic-RAG assistant: chunk local docs, index them with TF-IDF, and let Claude search them.

Usage: python agent.py [docs-folder]    (default: ./docs)
Try:   "why does my model do great on training data but badly on new data?"
"""
import json
import sys
from pathlib import Path

import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "claude-opus-5-5"
MAX_STEPS = 10  # safety cap on tool-use turns per question
CHUNK_WORDS, OVERLAP = 80, 20

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM = (
    "You answer questions using only the user's documents. Call search_docs first, and search again "
    "with a different query if the passages don't cover the question. Cite sources as [file#chunk]. "
    "If the documents don't contain the answer, say so instead of guessing."
)

# --- Ingest: split every .md/.txt file into overlapping word windows -------------------------------
DOCS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
CHUNKS: list[tuple[str, str]] = []  # (source file, chunk text)
for path in sorted(DOCS.glob("**/*")):
    if path.suffix in {".md", ".txt"}:
        words = path.read_text(encoding="utf-8", errors="replace").split()
        for start in range(0, max(len(words), 1), CHUNK_WORDS - OVERLAP):
            CHUNKS.append((path.name, " ".join(words[start : start + CHUNK_WORDS])))
if not CHUNKS:
    sys.exit(f"No .md or .txt files found in {DOCS}")

# --- Index: TF-IDF keeps retrieval local, fast and tiny (no embedding model or vector DB) ----------
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
MATRIX = VECTORIZER.fit_transform(text for _, text in CHUNKS)


def search_docs(query: str, k: int = 3) -> str:
    """Retrieve the top-k chunks by cosine similarity to the query."""
    scores = cosine_similarity(VECTORIZER.transform([query]), MATRIX).ravel()
    top = [i for i in scores.argsort()[::-1][:k] if scores[i] > 0]
    return "\n\n".join(
        f"[{CHUNKS[i][0]}#{i}] (score {scores[i]:.2f})\n{CHUNKS[i][1]}" for i in top
    ) or "No relevant passages found."


TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the document collection and return the most relevant passages with their sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or a short question"},
                "k": {"type": "integer", "description": "Number of passages, default 3"},
            },
            "required": ["query"],
        },
    }
]


def answer(messages: list) -> str:
    """The agent loop: Claude decides when and what to retrieve, and can search several times."""
    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [search] {json.dumps(block.input)}")
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": search_docs(**block.input)}
                )
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached."


if __name__ == "__main__":
    history: list = []  # kept across questions for follow-ups
    print(f"Indexed {len(CHUNKS)} chunks from {DOCS} (Ctrl+C to quit)")
    while True:
        question = input("\nyou> ").strip()
        if question:
            history.append({"role": "user", "content": question})
            print("\n" + answer(history))
