"""
Naive RAG — query. Phase 1, Track B.
Query -> Embed -> Cosine search (top-k) -> Generate with local LLM.

Run:  python query.py "How does the Jira cursor-ai automation work?"
Needs: store.json from ingest.py, ollama serve running.
"""
import json
import sys
from pathlib import Path

import numpy as np
import ollama

from config import EMBED_MODEL, LLM_MODEL, STORE_FILE, TOP_K

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question"')
        sys.exit(1)
    if not Path(STORE_FILE).exists():
        print(f"Missing {STORE_FILE} — run python ingest.py first.")
        sys.exit(1)

    question = sys.argv[1]
    store = json.loads(Path(STORE_FILE).read_text(encoding="utf-8"))
    q_emb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"])

    scored = []
    for item in store:
        score = cosine(q_emb, np.array(item["embedding"]))
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]

    print("Retrieved:")
    for score, item in top:
        print(f"  {score:.3f}  {item['id']}")

    context = "\n\n".join(f"[{item['source']}] {item['text']}" for _, item in top)
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer from the context below. You may compare and combine facts from different documents to answer. Use only facts present in the context, do not invent new facts. If the context truly contains no relevant facts, say 'I don't know from these documents.' Cite the source file name.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    print(f"\nAnswer:\n{resp['message']['content']}")
    print(f"\nSources: {', '.join(sorted({item['source'] for _, item in top}))}")


if __name__ == "__main__":
    main()
