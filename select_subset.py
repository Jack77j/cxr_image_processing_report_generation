"""
Step 1b (optional) - Pick a smaller, better-balanced training set with a fixed IMAGE budget.

    python select_subset.py --images 25000
    python select_subset.py --images 25000 --images_per_study 2     # frontal + lateral pairs

Reads  outputs/processed/train.csv   (from prepare_data.py; val.csv and test.csv are NOT touched)
Writes outputs/processed/train_subset_<N>k.csv      -> pass to train.py with --train_csv
       (train_subset_<N>k_2img.csv when --images_per_study 2)
       outputs/processed/train_subset_<N>_summary.json
       outputs/figures/train_subset_<N>_prevalence.png

How the studies are chosen
  1. Quality filters
       * the study must contain a frontal image (PA or AP)
       * the cleaned report must be SUBSET_MIN_WORDS..SUBSET_MAX_WORDS words long
         (drops one-line and runaway reports)
  2. Patient diversity: at most SUBSET_MAX_PER_PATIENT studies per patient, so the subset
     covers many different patients instead of a few heavily imaged ones.
  3. Finding balance (the main lever against the "normal report for everyone" problem)
       * every report is labelled with the 14 CheXpert findings (clinical_labeler.py)
       * each study gets a sampling weight  (1 / frequency of its rarest finding) ** SUBSET_BALANCE_ALPHA
         so studies with rare findings (pneumothorax, consolidation, lesions, ...) are picked more often
       * normal studies ("No Finding") are capped at SUBSET_NORMAL_MAX_FRAC of the subset
       * reports taken from the Findings section get full weight, reports that fell back to the
         Impression get SUBSET_IMPRESSION_WEIGHT (Findings text is more descriptive)
  4. Image budget: images are counted exactly; with images_per_study = 1 each study contributes
     its first frontal image, with 2 it contributes frontal + lateral when available.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from clinical_labeler import OBSERVATIONS, label_reports

FRONTAL = ("PA", "AP")
LATERAL = ("LATERAL", "OTHER")      # "OTHER" in this dataset is almost always the left-lateral (LL) view


def choose_images(paths, views, per_study):
    """Frontal first; with per_study >= 2 add a lateral (else a second frontal)."""
    frontal = [p for p, v in zip(paths, views) if v in FRONTAL]
    lateral = [p for p, v in zip(paths, views) if v in LATERAL]
    if not frontal:
        return [], []
    chosen = [frontal[0]]
    extra = lateral + frontal[1:]
    chosen += extra[: max(0, per_study - 1)]
    view_of = dict(zip(paths, views))
    return chosen, [view_of[p] for p in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=config.SUBSET_IMAGES or 25000, help="total training images")
    ap.add_argument("--images_per_study", type=int, default=config.SUBSET_IMAGES_PER_STUDY, choices=[1, 2])
    ap.add_argument("--train_csv", default=os.path.join(config.PROCESSED_DIR, "train.csv"))
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--min_words", type=int, default=config.SUBSET_MIN_WORDS)
    ap.add_argument("--max_words", type=int, default=config.SUBSET_MAX_WORDS)
    ap.add_argument("--max_per_patient", type=int, default=config.SUBSET_MAX_PER_PATIENT)
    ap.add_argument("--normal_max_frac", type=float, default=config.SUBSET_NORMAL_MAX_FRAC)
    ap.add_argument("--balance_alpha", type=float, default=config.SUBSET_BALANCE_ALPHA)
    ap.add_argument("--impression_weight", type=float, default=config.SUBSET_IMPRESSION_WEIGHT)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    tag = f"train_subset_{args.images // 1000}k" if args.images % 1000 == 0 else f"train_subset_{args.images}"
    if args.images_per_study == 2:
        tag += "_2img"
    out_csv = args.out_csv or os.path.join(os.path.dirname(args.train_csv), f"{tag}.csv")

    df = pd.read_csv(args.train_csv)
    df["image_paths"] = df["image_paths"].map(json.loads)
    df["views"] = df["views"].map(json.loads)
    n_all = len(df)
    print(f"Loaded {n_all:,} training studies / {int(df['n_images'].sum()):,} images from {args.train_csv}")

    # ---------------- 1. quality filters ----------------
    picked = [choose_images(p, v, args.images_per_study) for p, v in zip(df["image_paths"], df["views"])]
    df["sel_paths"] = [p for p, _ in picked]
    df["sel_views"] = [v for _, v in picked]
    df["n_sel"] = df["sel_paths"].map(len)
    words = df["target"].astype(str).str.split().str.len()
    keep = (df["n_sel"] > 0) & words.between(args.min_words, args.max_words)
    filt = {"no_frontal": int((df["n_sel"] == 0).sum()),
            "outside_word_range": int((~words.between(args.min_words, args.max_words)).sum())}
    df = df[keep].reset_index(drop=True)
    print(f"After quality filters: {len(df):,} studies  {filt}")

    # ---------------- 3. finding labels + sampling weights ----------------
    print("Labelling reports with the 14 CheXpert findings ...")
    labels = pd.DataFrame(label_reports(df["target"].tolist()))[OBSERVATIONS]
    freq = labels.mean().clip(lower=1e-4)
    abnormal = [o for o in OBSERVATIONS if o != "No Finding"]
    inv = (1.0 / freq) ** args.balance_alpha
    pos = labels[abnormal].values.astype(bool)
    rarest = np.where(pos, inv[abnormal].values[None, :], 0.0).max(axis=1)
    is_normal = labels["No Finding"].values == 1      # no abnormality (support devices alone count as normal)
    weight = np.where(is_normal, inv["No Finding"], rarest)
    weight = weight * np.where(df["source_section"].values == "findings", 1.0, args.impression_weight)

    # weighted sampling without replacement (Efraimidis-Spirakis keys), then greedy fill under the caps
    rng = np.random.default_rng(args.seed)
    keys = rng.random(len(df)) ** (1.0 / np.maximum(weight, 1e-12))
    order = np.argsort(-keys)

    per_patient, n_images, n_normal, chosen = {}, 0, 0, []
    max_normal_images = args.normal_max_frac * args.images
    for i in order:
        if n_images >= args.images:
            break
        sid = df.at[i, "subject_id"]
        if per_patient.get(sid, 0) >= args.max_per_patient:
            continue
        k = int(df.at[i, "n_sel"])
        if n_images + k > args.images:
            continue
        if is_normal[i] and n_normal + k > max_normal_images:
            continue
        chosen.append(i)
        per_patient[sid] = per_patient.get(sid, 0) + 1
        n_images += k
        n_normal += k if is_normal[i] else 0

    sub = df.loc[chosen].copy()
    sub["image_paths"] = sub["sel_paths"].map(json.dumps)
    sub["views"] = sub["sel_views"].map(json.dumps)
    sub["n_images"] = sub["n_sel"]
    sub["has_frontal"] = True
    cols = ["subject_id", "study_id", "image_paths", "views", "n_images", "has_frontal",
            "source_section", "n_dropped_sentences", "target", "target_augment", "report_raw"]
    sub = sub[cols].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sub.to_csv(out_csv, index=False)

    # ---------------- summary ----------------
    full_prev = (labels.mean() * 100).round(2)
    sub_prev = (labels.loc[chosen].mean() * 100).round(2)
    summary = {
        "settings": vars(args),
        "output_csv": out_csv,
        "input": {"studies": n_all},
        "after_quality_filters": {"studies": int(len(df)), "removed": filt},
        "subset": {
            "studies": int(len(sub)),
            "images": int(sub["n_images"].sum()),
            "patients": int(sub["subject_id"].nunique()),
            "normal_studies_pct": round(100 * float(is_normal[chosen].mean()), 1),
            "target_from_findings_pct": round(100 * float((sub["source_section"] == "findings").mean()), 1),
            "target_words_median": float(sub["target"].str.split().str.len().median()),
        },
        "finding_prevalence_pct": {o: {"filtered_pool": float(full_prev[o]), "subset": float(sub_prev[o])}
                                   for o in OBSERVATIONS},
    }
    with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["subset"], indent=2))
    print(pd.DataFrame({"filtered pool %": full_prev, "subset %": sub_prev}).to_string())
    print(f"Saved {out_csv}")

    # ---------------- figure: prevalence before vs after ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ink2, surface, grid = "#52514e", "#fcfcfb", "#e4e3df"
        order_obs = full_prev.sort_values(ascending=True).index.tolist()
        y = np.arange(len(order_obs))
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.set_facecolor(surface)
        ax.set_facecolor(surface)
        ax.barh(y + 0.2, [full_prev[o] for o in order_obs], height=0.38, color="#2a78d6",
                edgecolor=surface, linewidth=2, label="all training studies (after filters)")
        ax.barh(y - 0.2, [sub_prev[o] for o in order_obs], height=0.38, color="#eb6834",
                edgecolor=surface, linewidth=2, label=f"selected subset ({int(sub['n_images'].sum()):,} images)")
        ax.set_yticks(y, order_obs)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9, length=0)
        ax.set_xlabel("% of studies mentioning the finding", color=ink2)
        ax.set_title("Finding prevalence: full training set vs balanced subset", loc="left",
                     fontsize=12, fontweight="semibold", color="#0b0b0b")
        ax.legend(frameon=False, fontsize=9, labelcolor=ink2, loc="lower right")
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(config.FIGURES_DIR, f"{tag}_prevalence.png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200, facecolor=surface)
        print(f"Saved {fig_path}")
    except Exception as e:  # plotting is optional
        print(f"[warn] figure skipped: {e}")


if __name__ == "__main__":
    main()
