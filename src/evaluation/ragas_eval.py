""RAGAS evaluation: genera las métricas que van en el README."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.retrieval.hybrid import HybridRetriever
from src.api.main import generate_answer
import os


def run_eval(testset_path: str, output_dir: str) -> dict:
    with open(testset_path) as f:
        testset = json.load(f)  # [{question, ground_truth}]

    retriever = HybridRetriever(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection=os.getenv("QDRANT_COLLECTION", "documents"),
    )

    rows = []
    for item in testset:
        q = item["question"]
        chunks = retriever.retrieve(q, k=5)
        contexts = [c.text for c in chunks]
        answer = generate_answer(q, chunks)
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(testset),
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / "ragas_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", default="./outputs/")
    args = parser.parse_args()
    run_eval(args.testset, args.output)
