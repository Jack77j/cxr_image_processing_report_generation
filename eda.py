"""
Step 2 (optional) - Exploratory analysis of the processed study-level data.
Produces figures for the report / review slides.

    python eda.py                    # uses outputs/processed/train.csv
    python eda.py --split test

Figures (outputs/figures/<split>_*.png):
    images_per_study, view_distribution, target_length, target_source, finding_prevalence
plus <split>_eda_summary.csv with the numbers behind every figure.
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from clinical_labeler import OBSERVATIONS, label_reports

INK, INK_2, SURFACE, GRID, SERIES = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#2a78d6"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)


def bar(ax, labels, values, horizontal=False, fmt="{:,}"):
    if horizontal:
        bars = ax.barh(labels, values, color=SERIES, height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.invert_yaxis()
        for b, v in zip(bars, values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", ha="left", color=INK_2, fontsize=8)
    else:
        bars = ax.bar(labels, values, color=SERIES, width=0.7, edgecolor=SURFACE, linewidth=2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", color=INK_2, fontsize=8)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--processed_dir", default=config.PROCESSED_DIR)
    ap.add_argument("--out_dir", default=config.FIGURES_DIR)
    ap.add_argument("--max_studies", type=int, default=50000, help="sample for the labeler (speed)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.processed_dir, f"{args.split}.csv"))
    views = df["views"].map(json.loads)
    prefix = os.path.join(args.out_dir, args.split)
    summary = []

    # 1. images per study
    counts = df["n_images"].clip(upper=5).value_counts().sort_index()
    labels = [str(i) if i < 5 else "5+" for i in counts.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bar(ax, labels, counts.values.tolist())
    style(ax, f"X-ray images per study ({args.split}, n={len(df):,} studies)", "images in study", "studies")
    save(fig, f"{prefix}_images_per_study.png")
    summary += [("images_per_study", l, int(v)) for l, v in zip(labels, counts.values)]

    # 2. view distribution (all images)
    vc = Counter(v for vs in views for v in vs)
    order = [v for v in ("PA", "AP", "LATERAL", "OTHER") if v in vc]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bar(ax, order, [vc[v] for v in order])
    style(ax, f"View position of all images ({args.split})", "view", "images")
    save(fig, f"{prefix}_view_distribution.png")
    summary += [("view", v, int(vc[v])) for v in order]

    # 3. target length
    words = df["target"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(words.clip(upper=150), bins=30, color=SERIES, edgecolor=SURFACE, linewidth=2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axvline(words.median(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(words.median(), ax.get_ylim()[1] * 0.95, f"  median {words.median():.0f} words",
            color=INK_2, fontsize=8, va="top")
    style(ax, f"Cleaned target length ({args.split}, capped at 150 words)", "words", "studies")
    save(fig, f"{prefix}_target_length.png")
    summary += [("target_words", "median", float(words.median())),
                ("target_words", "p95", float(words.quantile(0.95)))]

    # 4. which section the target came from
    src = df["source_section"].fillna("").replace({"": "none"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.0))
    bar(ax, src.index.tolist(), src.values.tolist(), horizontal=True)
    style(ax, "Target taken from", "studies", "")
    save(fig, f"{prefix}_target_source.png")
    summary += [("target_source", k, int(v)) for k, v in src.items()]

    # 5. finding prevalence (rule-based labels on the targets)
    sample = df.sample(n=min(args.max_studies, len(df)), random_state=config.SEED)
    labels_df = pd.DataFrame(label_reports(sample["target"].tolist()))[OBSERVATIONS]
    prev = (labels_df.mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, prev.index.tolist(), prev.round(1).tolist(), horizontal=True, fmt="{:.1f}%")
    style(ax, f"Findings mentioned as present (% of {len(sample):,} studies)", "% of studies", "")
    save(fig, f"{prefix}_finding_prevalence.png")
    summary += [("finding_prevalence_pct", k, float(v)) for k, v in prev.items()]

    out = f"{prefix}_eda_summary.csv"
    pd.DataFrame(summary, columns=["figure", "category", "value"]).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
