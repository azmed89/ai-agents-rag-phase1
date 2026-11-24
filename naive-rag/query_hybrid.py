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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ollama

from config import EMBED_MODEL, STORE_FILE, TOP_K
from query import Chunk, synthesize

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
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


@dataclass
class HybridResult:
    top: list[tuple[Chunk, float, float]]
    max_dense: float
    dense_top: list[tuple[str, float]]
    bm25_top: list[tuple[str, float]]


def retrieve_hybrid(question: str, store: list[Chunk]) -> HybridResult:
    if not store:
        return HybridResult(top=[], max_dense=0.0, dense_top=[], bm25_top=[])

    # Dense ranking
    q_emb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"])
    dense_scores = [cosine(q_emb, np.array(item["embedding"])) for item in store]
    dense_rank = sorted(range(len(store)), key=lambda i: dense_scores[i], reverse=True)

    # BM25 ranking
    bm25 = BM25([tokenize(item["text"]) for item in store])
    bm25_scores = bm25.score(tokenize(question))
    bm25_rank = sorted(range(len(store)), key=lambda i: bm25_scores[i], reverse=True)

    # Fusion, then top-k
    fused = rrf([dense_rank, bm25_rank], k=60)[:TOP_K]
    return HybridResult(
        top=[(store[i], dense_scores[i], bm25_scores[i]) for i in fused],
        max_dense=dense_scores[dense_rank[0]],
        dense_top=[(store[i]["id"], dense_scores[i]) for i in dense_rank[:3]],
        bm25_top=[(store[i]["id"], bm25_scores[i]) for i in bm25_rank[:3]],
    )


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query_hybrid.py "your question"')
        sys.exit(1)
    question = sys.argv[1]
    store = json.loads(Path(STORE_FILE).read_text(encoding="utf-8"))
    result = retrieve_hybrid(question, store)

    print("Dense top-3:", [(chunk_id, round(score, 3)) for chunk_id, score in result.dense_top])
    print("BM25  top-3:", [(chunk_id, round(score, 2)) for chunk_id, score in result.bm25_top])
    print("Fused top-5:")
    for item, ds, bs in result.top:
        print(f"  {item['id']}  dense={ds:.3f}  bm25={bs:.2f}")

    chunks = [item for item, _, _ in result.top]
    print(f"\nAnswer:\n{synthesize(question, chunks)}")
    print(f"\nSources: {', '.join(sorted({item['source'] for item in chunks}))}")

if __name__ == "__main__":
    main()
