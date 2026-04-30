"""
Evaluate the RAG pipeline across multiple configurations.

Usage:
    uv run python -m evaluate.run_eval
    uv run python -m evaluate.run_eval --dry-run
    uv run python -m evaluate.run_eval --config k10_t0.4
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import get_qdrant_client
from evaluate.metrics import (
    hit,
    precision_at_k,
    reciprocal_rank,
    score_faithfulness,
    score_relevance,
)
from generating.generate import generate
from retrieving.retrieve import retrieve
from retrieving.bm25_retrieve import retrieve_bm25
from retrieving.hybrid_retrieve import retrieve_hybrid

load_dotenv()

BENCHMARK_PATH = Path("benchmarks/eval_questions.json")
RESULTS_DIR = Path("results")


@dataclass
class EvalConfig:
    name: str
    k: int
    score_threshold: float
    embedding_model: str = "intfloat/multilingual-e5-small"
    collection_name: str = "documents"
    retrieval_strategy: str = "dense"   # dense | bm25 | hybrid
    llm_model: str = "gpt-4.1-nano"

    def __str__(self) -> str:
        return (
            f"{self.name} "
            f"(k={self.k}, thr={self.score_threshold}, "
            f"ret={self.retrieval_strategy}, llm={self.llm_model})"
        )


# Baseline
_BASELINE = dict(k=10, score_threshold=0.4, embedding_model="intfloat/multilingual-e5-small",
                 collection_name="documents", retrieval_strategy="dense", llm_model="gpt-4.1-nano")

CONFIGS: list[EvalConfig] = [
    # Baseline
    EvalConfig(name="baseline",   **_BASELINE),

    # Retrieval strategy ablation
    EvalConfig(name="bm25",    k=10, score_threshold=0.4, retrieval_strategy="bm25"),
    EvalConfig(name="hybrid",  k=10, score_threshold=0.4, retrieval_strategy="hybrid"),

    # Chunking ablation
    EvalConfig(name="chunk-256",  k=10, score_threshold=0.4, collection_name="archelec_fixed_256"),
    EvalConfig(name="chunk-1024", k=10, score_threshold=0.4, collection_name="archelec_fixed_1024"),

    # Embedding ablation
    EvalConfig(name="e5-large", k=10, score_threshold=0.4,
               embedding_model="intfloat/multilingual-e5-large",
               collection_name="archelec_e5large_512"),

    # LLM ablation
    EvalConfig(name="gpt-4o-mini", k=10, score_threshold=0.4, llm_model="gpt-4o-mini"),
]


def _do_retrieve(config: EvalConfig, query: str, qdrant_client):
    if config.retrieval_strategy == "bm25":
        return retrieve_bm25(query, qdrant_client, config.collection_name, k=config.k)
    if config.retrieval_strategy == "hybrid":
        return retrieve_hybrid(
            query, qdrant_client, config.collection_name,
            k=config.k, model_name=config.embedding_model,
            score_threshold=config.score_threshold,
        )
    return retrieve(
        query=query,
        client=qdrant_client,
        collection_name=config.collection_name,
        k=config.k,
        model_name=config.embedding_model,
        score_threshold=config.score_threshold,
    )


def _run_one(config: EvalConfig, benchmark: list[dict], openai_client: OpenAI, qdrant_client, dry_run: bool) -> dict:
    per_question = []

    for q in benchmark:
        results = _do_retrieve(config, q["question"], qdrant_client)

        filenames = [r.filename for r in results]
        source_doc_id = q["source_doc_id"]

        h = hit(filenames, source_doc_id)
        rr = reciprocal_rank(filenames, source_doc_id)
        prec = precision_at_k(filenames, source_doc_id)

        if dry_run or not results:
            answer = ""
            faithfulness = 0.5
            relevance = 0.5
        else:
            answer = generate(q["question"], results, model=config.llm_model)
            context = "\n\n".join(r.chunk for r in results[:5])
            faithfulness = score_faithfulness(openai_client, q["question"], context, answer)
            relevance = score_relevance(openai_client, q["question"], answer)

        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "annee": q.get("annee", ""),
                "theme": q.get("theme", ""),
                "type": q.get("type", "named"),
                "source_doc_id": source_doc_id,
                "n_retrieved": len(results),
                "hit": h,
                "reciprocal_rank": rr,
                "precision": prec,
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "generated_answer": answer,
            }
        )

        flag = "✓" if h else "✗"
        print(
            f"    {flag} {q['id']}  rr={rr:.2f}  prec={prec:.2f}  "
            f"faith={faithfulness:.2f}  rel={relevance:.2f}  n={len(results)}"
        )

    n = len(per_question)
    return {
        "config": asdict(config),
        "n_questions": n,
        "hit_rate": sum(r["hit"] for r in per_question) / n,
        "mrr": sum(r["reciprocal_rank"] for r in per_question) / n,
        "mean_precision": sum(r["precision"] for r in per_question) / n,
        "mean_faithfulness": sum(r["faithfulness"] for r in per_question) / n,
        "mean_answer_relevance": sum(r["answer_relevance"] for r in per_question) / n,
        "per_question": per_question,
    }


def main(dry_run: bool = False, target: str | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    if not BENCHMARK_PATH.exists():
        print(f"Benchmark not found at {BENCHMARK_PATH}.")
        print("Run:  uv run python -m evaluate.benchmark_gen")
        sys.exit(1)

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} evaluation questions")

    qdrant_client = get_qdrant_client()
    openai_client = OpenAI()

    configs = [c for c in CONFIGS if target is None or c.name == target]
    if not configs:
        print(f"Unknown config: {target}")
        sys.exit(1)

    all_results = []
    results_path = RESULTS_DIR / "all_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)

    existing_names = {r["config"]["name"] for r in all_results}

    for config in configs:
        if config.name in existing_names and target is None:
            print(f"\nSkipping '{config.name}' (already in results)")
            continue

        print(f"\n  {config}")

        try:
            result = _run_one(config, benchmark, openai_client, qdrant_client, dry_run)
        except Exception as e:
            print(f"  [ERROR] {e} — skipping")
            continue

        # Replace if re-running a specific config
        all_results = [r for r in all_results if r["config"]["name"] != config.name]
        all_results.append(result)

        out = RESULTS_DIR / f"eval_{config.name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    w = 14
    header = f"{'Config':<{w}} {'Hit Rate':>9} {'MRR':>7} {'Precision':>10} {'Faithful':>9} {'Relevance':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in sorted(all_results, key=lambda x: x["config"]["name"]):
        name = r["config"]["name"]
        print(
            f"{name:<{w}} {r['hit_rate']:>9.3f} {r['mrr']:>7.3f} "
            f"{r['mean_precision']:>10.3f} {r['mean_faithfulness']:>9.3f} "
            f"{r['mean_answer_relevance']:>10.3f}"
        )
    print(f"\n✓ Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", help="Run only this config by name")
    args = parser.parse_args()
    main(dry_run=args.dry_run, target=args.config)
