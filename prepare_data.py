"""
Step 1 - Build study-level training data from the patient-level MIMIC-CXR CSVs.

Input  : mimic_cxr_aug_train.csv / mimic_cxr_aug_validate.csv  (one row per PATIENT)
Output : outputs/processed/{train,val,test}.csv                  (one row per STUDY)

For every patient row:
  * the stringified list columns are parsed,
  * images are grouped by study id (the 's5xxxxxxx' folder in the path),
  * the i-th unique study is paired with the i-th report in `text` / `text_augment`,
  * every image gets a view label from the AP / PA / Lateral columns,
  * each study's report is cleaned into a single-study target (text_cleaning.py),
  * images are ordered frontal-first (PA, AP, LATERAL, OTHER) for later selection.

Splits (always by subject_id, so no patient appears in two splits):
  train / val : from the TRAIN csv (val = VAL_FRACTION of patients)
  test        : the official VALIDATE csv (500 patients), never used for training

Usage:
    python prepare_data.py
    python prepare_data.py --limit_patients 2000      # quick dry run
"""
import argparse
import ast
import json
import os
import random
import re
from collections import Counter, OrderedDict

import pandas as pd
from tqdm.auto import tqdm

import config
from text_cleaning import build_target

VIEW_PRIORITY = {"PA": 0, "AP": 1, "LATERAL": 2, "OTHER": 3}
_STUDY_RE = re.compile(r"/(s\d+)/")


def parse_list(value):
    """Turn "['a', 'b']" into ['a', 'b']; NaN / malformed -> []."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() in ("", "[]"):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # some 'view' cells contain a bare nan: ['LL', nan]
        try:
            parsed = ast.literal_eval(value.replace("nan", "None"))
        except (ValueError, SyntaxError):
            return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def study_id_of(path):
    m = _STUDY_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else None


def patient_to_studies(row, section, comparison_mode):
    """Convert one patient row into a list of study dicts. Returns (studies, mismatch_flag)."""
    images = parse_list(row["image"])
    texts = parse_list(row["text"])
    texts_aug = parse_list(row["text_augment"])

    view_of = {}
    for col, label in (("PA", "PA"), ("AP", "AP"), ("Lateral", "LATERAL")):
        for p in parse_list(row.get(col, "[]")):
            view_of.setdefault(p, label)

    groups = OrderedDict()
    for p in images:
        sid = study_id_of(p)
        if sid is not None:
            groups.setdefault(sid, []).append(p)

    study_ids = list(groups.keys())
    mismatch = len(study_ids) != len(texts)
    studies = []
    for i, sid in enumerate(study_ids[: len(texts)]):
        raw = texts[i]
        raw_aug = texts_aug[i] if i < len(texts_aug) else raw
        cleaned = build_target(raw, section, comparison_mode)
        cleaned_aug = build_target(raw_aug, section, comparison_mode)

        paths = sorted(groups[sid], key=lambda p: VIEW_PRIORITY[view_of.get(p, "OTHER")])
        studies.append({
            "subject_id": int(row["subject_id"]),
            "study_id": sid,
            "image_paths": paths,
            "views": [view_of.get(p, "OTHER") for p in paths],
            "target": cleaned["target"],
            "target_augment": cleaned_aug["target"] or cleaned["target"],
            "source_section": cleaned["source_section"],
            "n_dropped_sentences": cleaned["n_dropped"],
            "report_raw": raw,
        })
    return studies, mismatch


def build_studies(csv_path, args, desc):
    df = pd.read_csv(csv_path)
    if args.limit_patients:
        df = df.head(args.limit_patients)
    all_studies, mismatches = [], 0
    for row in tqdm(df.to_dict("records"), desc=f"Parsing {desc}"):
        studies, mismatch = patient_to_studies(row, args.section, args.comparison_mode)
        mismatches += int(mismatch)
        all_studies.extend(studies)
    return all_studies, mismatches, len(df)


def filter_studies(studies, args, desc):
    stats = Counter()
    kept = []
    for s in tqdm(studies, desc=f"Filtering {desc}"):
        if args.check_images:
            exists = [os.path.isfile(os.path.join(args.image_root, p)) for p in s["image_paths"]]
            stats["missing_images"] += exists.count(False)
            s["views"] = [v for v, e in zip(s["views"], exists) if e]
            s["image_paths"] = [p for p, e in zip(s["image_paths"], exists) if e]
        if not s["image_paths"]:
            stats["dropped_no_image"] += 1
            continue
        if len(s["target"].split()) < args.min_words:
            stats["dropped_empty_target"] += 1
            continue
        s["n_images"] = len(s["image_paths"])
        s["has_frontal"] = any(v in ("PA", "AP") for v in s["views"])
        kept.append(s)
    stats["kept"] = len(kept)
    return kept, dict(stats)


COLUMNS = ["subject_id", "study_id", "image_paths", "views", "n_images", "has_frontal",
           "source_section", "n_dropped_sentences", "target", "target_augment", "report_raw"]


def to_frame(studies):
    if not studies:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(studies)
    df["image_paths"] = df["image_paths"].map(json.dumps)
    df["views"] = df["views"].map(json.dumps)
    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_csv", default=config.RAW_TRAIN_CSV)
    ap.add_argument("--test_csv", default=config.RAW_VAL_CSV)
    ap.add_argument("--image_root", default=config.IMAGE_ROOT)
    ap.add_argument("--out_dir", default=config.PROCESSED_DIR)
    ap.add_argument("--section", default=config.TARGET_SECTION,
                    choices=["findings_fallback_impression", "findings_only", "findings_impression"])
    ap.add_argument("--comparison_mode", default=config.COMPARISON_MODE, choices=["drop", "keep"])
    ap.add_argument("--val_fraction", type=float, default=config.VAL_FRACTION)
    ap.add_argument("--min_words", type=int, default=config.MIN_TARGET_WORDS)
    ap.add_argument("--no_check_images", dest="check_images", action="store_false",
                    default=config.CHECK_IMAGES_EXIST)
    ap.add_argument("--limit_patients", type=int, default=None, help="debug: only read first N patients")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- parse ----------
    train_studies, train_mm, n_train_pat = build_studies(args.train_csv, args, "train csv")
    test_studies, test_mm, n_test_pat = build_studies(args.test_csv, args, "test csv")

    # ---------- no patient may appear in both train and test ----------
    test_subjects = {s["subject_id"] for s in test_studies}
    overlap = {s["subject_id"] for s in train_studies} & test_subjects
    if overlap:
        print(f"[warn] {len(overlap)} subjects appear in both CSVs -> removed from train.")
        train_studies = [s for s in train_studies if s["subject_id"] not in overlap]

    # ---------- filter ----------
    train_studies, train_stats = filter_studies(train_studies, args, "train")
    test_studies, test_stats = filter_studies(test_studies, args, "test")

    for name, st in (("train", train_stats), ("test", test_stats)):
        if args.check_images and st.get("dropped_no_image", 0) > 0.5 * (st["kept"] + st.get("dropped_no_image", 0)):
            print(f"[WARNING] most {name} studies have no image on disk - is --image_root correct?\n"
                  f"          image_root = {args.image_root}")

    # ---------- split train into train / val by subject ----------
    subjects = sorted({s["subject_id"] for s in train_studies})
    random.Random(args.seed).shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * args.val_fraction)))
    val_subjects = set(subjects[:n_val])
    val_studies = [s for s in train_studies if s["subject_id"] in val_subjects]
    train_studies = [s for s in train_studies if s["subject_id"] not in val_subjects]

    splits = {"train": train_studies, "val": val_studies, "test": test_studies}
    summary = {
        "settings": {k: v for k, v in vars(args).items()},
        "patients_in_csv": {"train_csv": n_train_pat, "test_csv": n_test_pat},
        "study_report_count_mismatches": {"train_csv": train_mm, "test_csv": test_mm},
        "filtering": {"train_csv": train_stats, "test_csv": test_stats},
        "splits": {},
    }
    for name, studies in splits.items():
        df = to_frame(studies)
        path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        words = df["target"].astype(str).str.split().str.len()
        summary["splits"][name] = {
            "patients": int(df["subject_id"].nunique()),
            "studies": int(len(df)),
            "images": int(df["n_images"].sum()),
            "studies_with_frontal": int(df["has_frontal"].astype(bool).sum()),
            "target_from_findings": int((df["source_section"] == "findings").sum()),
            "target_from_impression": int((df["source_section"] == "impression").sum()),
            "target_words_median": float(words.median()) if len(df) else 0.0,
            "target_words_p95": float(words.quantile(0.95)) if len(df) else 0.0,
        }
        print(f"Saved {path}  ({len(df):,} studies, {df['subject_id'].nunique():,} patients)")

    with open(os.path.join(args.out_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["splits"], indent=2))
    print(json.dumps(summary["filtering"], indent=2))


if __name__ == "__main__":
    main()
