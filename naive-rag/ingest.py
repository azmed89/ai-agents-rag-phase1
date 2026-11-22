"""
Naive RAG — ingest. Phase 1, Track B.
Ingest -> Clean -> Chunk -> Embed -> Store (in-memory JSON, no vector DB yet).

Run:  python ingest.py
Needs: ollama serve running, ollama pull nomic-embed-text
"""
import json
from pathlib import Path

import ollama

from config import CHUNK_WORDS, DOCS_DIR, EMBED_MODEL, OVERLAP_WORDS, STORE_FILE


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Recursive-ish chunking, simple version: split on paragraphs, then by words."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def main() -> None:
    docs_dir = Path(DOCS_DIR)
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        print(f"No .md files in {DOCS_DIR}/ — add your docs first.")
        return

    store = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
            store.append({"id": f"{f.name}#{i}", "source": f.name, "text": chunk, "embedding": emb})
        print(f"{f.name}: {len(chunks)} chunks")

    Path(STORE_FILE).write_text(json.dumps(store), encoding="utf-8")
    print(f"Saved {len(store)} chunks to {STORE_FILE}")


if __name__ == "__main__":
    main()
