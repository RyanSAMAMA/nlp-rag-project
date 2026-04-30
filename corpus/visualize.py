"""
Generate descriptive figures for the Archelec corpus.

Usage:
    uv run python -m corpus.visualize
    uv run python -m corpus.visualize --show
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

STATS_PATH = Path("results/corpus_stats.json")
FIGURES_DIR = Path("figures")

PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {"font.family": "serif", "savefig.dpi": 300, "savefig.bbox": "tight"}
    )


def _save(fig, name: str, show: bool):
    path = FIGURES_DIR / name
    fig.savefig(path)
    print(f"Saved at {path}")
    if show:
        plt.show()
    plt.close(fig)


# documents by year
def plot_by_year(stats: dict, show: bool = False):
    by_year = stats["by_year"]
    years = list(by_year.keys())
    counts = list(by_year.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        years, counts, color=PALETTE[: len(years)], width=0.5, edgecolor="white"
    )
    ax.bar_label(bars, fontsize=11, padding=3)
    ax.set_xlabel("Année d'élection")
    ax.set_ylabel("Nombre de candidats")
    ax.set_title("Distribution des manifestes par année")
    ax.set_ylim(0, max(counts) * 1.2)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_01_by_year.png", show)


# gender distribution by year
def plot_gender(stats: dict, show: bool = False):
    by_sexe_year = stats["by_sexe_year"]
    years = sorted(by_sexe_year.keys())
    genders = sorted({g for c in by_sexe_year.values() for g in c})
    gender_labels = {"M": "Homme", "F": "Femme"}

    x = np.arange(len(years))
    width = 0.3
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, g in enumerate(genders):
        vals = [by_sexe_year[y].get(g, 0) for y in years]
        offset = (i - len(genders) / 2) * width + width / 2
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=gender_labels.get(g, g),
            color=PALETTE[i],
            alpha=0.88,
        )
        ax.bar_label(bars, fontsize=10, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Nombre de candidats")
    ax.set_title("Répartition par genre et par année")
    ax.legend()
    ax.set_ylim(
        0, max(by_sexe_year[y].get(g, 0) for y in years for g in genders) * 1.25
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_02_gender.png", show)


# top parties
def plot_top_parties(stats: dict, top_n: int = 15, show: bool = False):
    parties = list(stats["by_parti"].items())[:top_n]
    labels = [p[0][:35] for p in parties]
    values = [p[1] for p in parties]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=sns.color_palette("muted", len(labels)),
        alpha=0.87,
    )
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xlabel("Nombre de candidats")
    ax.set_title(f"Top {top_n} partis / étiquettes politiques")
    ax.set_xlim(0, max(values) * 1.15)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_03_top_parties.png", show)


# top departments
def plot_top_departments(stats: dict, top_n: int = 20, show: bool = False):
    depts = list(stats["by_departement"].items())[:top_n]
    labels = [d[0][:30] for d in depts]
    values = [d[1] for d in depts]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=sns.color_palette("Blues_d", len(labels)),
        alpha=0.87,
    )
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xlabel("Nombre de candidats")
    ax.set_title(f"Top {top_n} départements")
    ax.set_xlim(0, max(values) * 1.15)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_04_top_departments.png", show)


# chunk length distribution
def plot_chunk_lengths(stats: dict, show: bool = False):
    lengths = stats["chunk_lengths"]["all"]
    mean = stats["chunk_lengths"]["mean"]
    median = stats["chunk_lengths"]["median"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, color=PALETTE[0], alpha=0.75, edgecolor="white")
    ax.axvline(
        mean, color=PALETTE[1], lw=1.8, linestyle="--", label=f"Moyenne ({mean:.0f})"
    )
    ax.axvline(
        median, color=PALETTE[2], lw=1.8, linestyle="-.", label=f"Médiane ({median})"
    )
    ax.set_xlabel("Longueur du chunk (caractères)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des longueurs de chunks")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_05_chunk_lengths.png", show)


# chunks per document
def plot_chunks_per_doc(stats: dict, show: bool = False):
    counts = stats["chunks_per_doc"]["all"]
    mean = stats["chunks_per_doc"]["mean"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        counts,
        bins=range(1, max(counts) + 2),
        color=PALETTE[3],
        alpha=0.78,
        edgecolor="white",
    )
    ax.axvline(
        mean, color=PALETTE[1], lw=1.8, linestyle="--", label=f"Moyenne ({mean:.1f})"
    )
    ax.set_xlabel("Nombre de chunks par manifeste")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution du nombre de chunks par document")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_06_chunks_per_doc.png", show)


# top profession
def plot_professions(stats: dict, top_n: int = 15, show: bool = False):
    profs = list(stats["by_profession"].items())[:top_n]
    if not profs:
        return
    labels = [p[0][:40] for p in profs]
    values = [p[1] for p in profs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=sns.color_palette("muted", len(labels)),
        alpha=0.87,
    )
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xlabel("Nombre de candidats")
    ax.set_title(f"Top {top_n} professions déclarées")
    ax.set_xlim(0, max(values) * 1.15)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "corpus_07_professions.png", show)


# Main
def main(show: bool = False):
    if not STATS_PATH.exists():
        print(f"Stats not found at {STATS_PATH}.")
        print("Run:  uv run python -m corpus.stats")
        return

    FIGURES_DIR.mkdir(exist_ok=True)
    _setup()

    with open(STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)

    print(f"Generating corpus figures\n")
    plot_by_year(stats, show)
    plot_gender(stats, show)
    plot_top_parties(stats, show=show)
    plot_top_departments(stats, show=show)
    plot_chunk_lengths(stats, show)
    plot_chunks_per_doc(stats, show)
    plot_professions(stats, show=show)
    print(f"\nAll corpus figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(show=args.show)
