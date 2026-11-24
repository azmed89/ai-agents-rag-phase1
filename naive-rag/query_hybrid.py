"""
Hybrid RAG — Phase 2, Session 2.
Query -> BM25 + dense cosine -> Reciprocal Rank Fusion -> Generate.
Run:  python query_hybrid.py "How does the Jira cursor-ai automation work?"
Needs: store.json from ingest.py, ollama serve running.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import ollama

from config import EMBED_MODEL, LLM_MODEL, STORE_FILE, TOP_K

# --- BM25, from scratch (no new dependency) ---
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())

class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.df = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1

    def score(self, query: list[str]) -> list[float]:
        out = []
        for d in self.docs:
            tf = Counter(d)
            s, dl = 0.0, len(d)
            for term in query:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

# --- Reciprocal Rank Fusion, k=60 (standard) ---
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query_hybrid.py "your question"')
        sys.exit(1)
    question = sys.argv[1]
    store = json.loads(Path(STORE_FILE).read_text(encoding="utf-8"))

    # Dense ranking
    q_emb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"])
    dense_scores = [cosine(q_emb, np.array(item["embedding"])) for item in store]
    dense_rank = sorted(range(len(store)), key=lambda i: dense_scores[i], reverse=True)

    # BM25 ranking
    bm25 = BM25([tokenize(item["text"]) for item in store])
    bm25_scores = bm25.score(tokenize(question))
    bm25_rank = sorted(range(len(store)), key=lambda i: bm25_scores[i], reverse=True)

    # Fusion, then top-k
    fused = rrf([dense_rank, bm25_rank])[:TOP_K]
    top = [(store[i], dense_scores[i], bm25_scores[i]) for i in fused]

    print("Dense top-3:", [(store[i]["id"], round(dense_scores[i], 3)) for i in dense_rank[:3]])
    print("BM25  top-3:", [(store[i]["id"], round(bm25_scores[i], 2)) for i in bm25_rank[:3]])
    print("Fused top-5:")
    for item, ds, bs in top:
        print(f"  {item['id']}  dense={ds:.3f}  bm25={bs:.2f}")

    context = "\n\n".join(f"[{item['source']}] {item['text']}" for item, _, _ in top)
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
    print(f"\nSources: {', '.join(sorted({item['source'] for item, _, _ in top}))}")

if __name__ == "__main__":
    main()