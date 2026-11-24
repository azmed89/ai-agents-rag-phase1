"""Run hybrid RAG with a LangGraph branch for low retrieval scores."""

import argparse
import json
from operator import add
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from config import STORE_FILE
from query import UNKNOWN_ANSWER, Chunk, synthesize
from query_hybrid import retrieve_hybrid

MIN_DENSE = 0.5
Route = Literal["generator", "unknown"]


class State(TypedDict, total=False):
    question: str
    chunks: list[Chunk]
    max_dense: float
    route: Route
    answer: str
    steps: Annotated[list[str], add]


def planner(state: State) -> State:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return {"question": question, "steps": ["planner"]}


def retriever(state: State) -> State:
    store_path = Path(__file__).resolve().parent / STORE_FILE
    store = json.loads(store_path.read_text(encoding="utf-8"))
    result = retrieve_hybrid(state["question"], store)
    return {
        "chunks": [chunk for chunk, _, _ in result.top],
        "max_dense": result.max_dense,
        "steps": ["retriever"],
    }


def select_route(state: State) -> Route:
    return "unknown" if state["max_dense"] < MIN_DENSE else "generator"


def generator(state: State) -> State:
    return {
        "route": "generator",
        "answer": synthesize(state["question"], state["chunks"]),
        "steps": ["generator"],
    }


def unknown(_state: State) -> State:
    return {"route": "unknown", "answer": UNKNOWN_ANSWER, "steps": ["unknown"]}


workflow = StateGraph(State)
workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("generator", generator)
workflow.add_node("unknown", unknown)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_conditional_edges("retriever", select_route)
workflow.add_edge("generator", END)
workflow.add_edge("unknown", END)
graph = workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question cannot be empty")
    try:
        result = graph.invoke({"question": args.question})
    except FileNotFoundError:
        parser.exit(1, f"Missing {STORE_FILE}. Run python ingest.py first.\n")

    sources = (
        sorted({chunk["source"] for chunk in result["chunks"]})
        if result["route"] == "generator"
        else []
    )
    print(f"Steps: {' -> '.join(result['steps'])}")
    print(f"Route: {result['route']}")
    print(f"Max dense score: {result['max_dense']:.6f}")
    print(f"Answer:\n{result['answer']}")
    print(f"Sources: {', '.join(sources) or '(none)'}")


if __name__ == "__main__":
    main()
