"""
Generate publication-quality evaluation figures using ablation-study layout.

Usage:
    uv run python -m evaluate.visualize
    uv run python -m evaluate.visualize --show
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_PATH = Path("results/all_results.json")
FIGURES_DIR = Path("figures")

BASELINE = "baseline"

# Ablation groups: title -> {configs, labels}
# labels override the default name per group context
ABLATION_GROUPS: dict[str, dict] = {
    "Retrieval Strategy": {
        "configs": ["baseline", "bm25", "hybrid"],
        "labels": {"baseline": "Dense", "bm25": "BM25", "hybrid": "Hybrid (RRF)"},
    },
    "Chunk Size": {
        "configs": ["chunk-256", "baseline", "chunk-1024"],
        "labels": {"chunk-256": "Fixed 256", "baseline": "Recursive 512", "chunk-1024": "Fixed 1024"},
    },
    "Embedding Model": {
        "configs": ["baseline", "e5-large"],
        "labels": {"baseline": "E5-Small (384d)", "e5-large": "E5-Large (1024d)"},
    },
    "Generation Model": {
        "configs": ["baseline", "gpt-4o-mini"],
        "labels": {"baseline": "GPT-4.1-nano", "gpt-4o-mini": "GPT-4o-mini"},
    },
}

# Metrics shown in ablation subplots
ABLATION_METRICS = {
    "hit_rate":           "Hit Rate",
    "mrr":                "MRR",
    "mean_answer_relevance": "Answer Relevance",
}

ALL_METRIC_KEYS = ["hit_rate", "mrr", "mean_precision", "mean_faithfulness", "mean_answer_relevance"]
ALL_METRIC_LABELS = {
    "hit_rate":              "Hit Rate",
    "mrr":                   "MRR",
    "mean_precision":        "Precision",
    "mean_faithfulness":     "Faithfulness",
    "mean_answer_relevance": "Answer Relevance",
}


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "savefig.bbox": "tight"})


def _save(fig, name: str, show: bool):
    path = FIGURES_DIR / name
    fig.savefig(path)
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def _index(results: list[dict]) -> dict[str, dict]:
    return {r["config"]["name"]: r for r in results}


# 1. Main ablation figure (2×2 grid)
def plot_ablation(results: list[dict], show: bool = False):
    idx = _index(results)
    metrics = list(ABLATION_METRICS.keys())
    metric_labels = list(ABLATION_METRICS.values())
    n_metrics = len(metrics)
    baseline_vals = {m: idx[BASELINE][m] for m in metrics} if BASELINE in idx else {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    group_palette = sns.color_palette("tab10", 4)

    for ax_idx, (group_title, group) in enumerate(ABLATION_GROUPS.items()):
        ax = axes[ax_idx]
        cfg_names = group["configs"]
        grp_labels = group["labels"]
        available = [c for c in cfg_names if c in idx]
        if not available:
            ax.set_visible(False)
            continue

        x = np.arange(len(available))
        width = 0.22
        metric_colors = sns.color_palette("muted", n_metrics)

        for m_idx, (metric, label) in enumerate(ABLATION_METRICS.items()):
            vals = [idx[c][metric] for c in available]
            offset = (m_idx - n_metrics / 2) * width + width / 2
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=metric_colors[m_idx], alpha=0.85, edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

        # Baseline reference lines
        if baseline_vals:
            for m_idx, metric in enumerate(metrics):
                ax.axhline(baseline_vals[metric],
                           color=metric_colors[m_idx], lw=1.2,
                           linestyle="--", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([grp_labels.get(c, c) for c in available], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(group_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Score")
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        sns.despine(ax=ax)

    fig.suptitle("Ablation Study — RAG Pipeline on Archelec", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "01_ablation.png", show)


# 2. Summary heatmap
def plot_heatmap(results: list[dict], show: bool = False):
    # Only show configs present in at least one ablation group
    shown = {c for grp in ABLATION_GROUPS.values() for c in grp["configs"]}
    results = [r for r in results if r["config"]["name"] in shown]
    if not results:
        return

    configs = [r["config"]["name"] for r in results]
    data = np.array([[r[k] for k in ALL_METRIC_KEYS] for r in results])

    fig, ax = plt.subplots(figsize=(10, max(3, len(configs) * 0.55 + 1.5)))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ALL_METRIC_KEYS)))
    ax.set_xticklabels([ALL_METRIC_LABELS[k] for k in ALL_METRIC_KEYS], rotation=18, ha="right")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)

    for i in range(len(configs)):
        for j in range(len(ALL_METRIC_KEYS)):
            color = "white" if data[i, j] > 0.65 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
    ax.set_title("Evaluation Summary Heatmap", pad=10)
    plt.tight_layout()
    _save(fig, "02_heatmap.png", show)


# 3. Score distributions (box plots) — retrieval ablation only
def plot_distributions(results: list[dict], show: bool = False):
    ret_configs = [c for c in ABLATION_GROUPS["Retrieval Strategy"]["configs"]
                   if c in _index(results)]
    if not ret_configs:
        return

    idx = _index(results)
    dist_metrics = [
        ("reciprocal_rank", "Reciprocal Rank"),
        ("precision",       "Precision @k"),
        ("faithfulness",    "Faithfulness"),
        ("answer_relevance","Answer Relevance"),
    ]

    fig, axes = plt.subplots(1, len(dist_metrics), figsize=(13, 4.5), sharey=True)
    palette = sns.color_palette("muted", len(ret_configs))

    for ax, (key, label) in zip(axes, dist_metrics):
        data = [[q[key] for q in idx[c]["per_question"]] for c in ret_configs]
        ret_labels = ABLATION_GROUPS["Retrieval Strategy"]["labels"]
        labels = [ret_labels.get(c, c) for c in ret_configs]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False,
                        medianprops={"color": "black", "lw": 1.5})
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(axis="x", labelsize=9)
        sns.despine(ax=ax, left=(ax != axes[0]))

    fig.suptitle("Score Distributions — Retrieval Strategy Comparison", y=1.02, size=12)
    plt.tight_layout()
    _save(fig, "03_distributions_retrieval.png", show)


# 4. Named vs Thematic hit rate — per ablation group
def plot_question_types(results: list[dict], show: bool = False):
    idx = _index(results)

    fig, axes = plt.subplots(1, len(ABLATION_GROUPS), figsize=(14, 4.5), sharey=True)
    types = ["named", "thematic"]
    type_labels = ["Nommée", "Thématique"]
    type_colors = sns.color_palette("Set2", 2)

    for ax, (group_title, group) in zip(axes, ABLATION_GROUPS.items()):
        grp_labels = group["labels"]
        available = [c for c in group["configs"] if c in idx]
        if not available:
            continue

        x = np.arange(len(available))
        width = 0.3

        for t_idx, (qtype, tlabel) in enumerate(zip(types, type_labels)):
            vals = []
            for c in available:
                per_q = idx[c]["per_question"]
                qs = [q for q in per_q if q.get("type", "named") == qtype]
                vals.append(sum(q["hit"] for q in qs) / len(qs) if qs else 0.0)
            offset = (t_idx - 1) * width + width / 2
            bars = ax.bar(x + offset, vals, width, label=tlabel,
                          color=type_colors[t_idx], alpha=0.85)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels([grp_labels.get(c, c) for c in available],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(group_title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.2)
        if ax == axes[0]:
            ax.set_ylabel("Hit Rate")
            ax.legend(fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle("Hit Rate by Question Type Across Ablation Groups", y=1.02, size=12)
    plt.tight_layout()
    _save(fig, "04_question_types.png", show)


# Main
def main(show: bool = False):
    if not RESULTS_PATH.exists():
        print(f"No results at {RESULTS_PATH}. Run: uv run python -m evaluate.run_eval")
        return

    FIGURES_DIR.mkdir(exist_ok=True)
    _setup()

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    available = [r["config"]["name"] for r in results]
    print(f"Loaded {len(results)} configs: {available}\n")

    plot_ablation(results, show)
    plot_heatmap(results, show)
    plot_distributions(results, show)
    plot_question_types(results, show)
    print(f"\n✓ All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(show=args.show)
