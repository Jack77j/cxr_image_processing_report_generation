"""
Score individual reports (one row per study) with the same metrics as evaluate.py.

evaluate.py gives one average over the whole test set. This script gives the score of each single
report, e.g. for the example reports shown in the presentation.

    # sample reports written by train.py after an epoch
    python score_examples.py --input outputs/checkpoints/subset25k_v1/samples_epoch5.txt

    # the same samples, read from the output of a Kaggle notebook (.ipynb) whose files
    # could not be downloaded (cell "training curve + latest sample reports")
    python score_examples.py --input results_with_image_processing.ipynb

    # any predictions file from generate.py, only some studies
    python score_examples.py --input outputs/results/subset25k_v1/predictions_test.csv --studies s56108500 s59470485

Accepted inputs
  * samples_*.txt from train.py            ("### <study>\\nREFERENCE: ...\\nGENERATED: ...")
  * a Kaggle notebook .ipynb that printed such a samples file
  * predictions_*.csv from generate.py     (columns study_id, reference, generated)

Per report it computes
  * BLEU-1 and BLEU-4  : NLTK sentence_bleu with the same tokenizer and smoothing as the corpus BLEU
                         in evaluate.py (sentence-level BLEU is the single-report version)
  * METEOR, ROUGE-L    : the same functions as evaluate.py
  * clinical metrics   : both reports are labelled with clinical_labeler.py (14 CheXpert observations);
                         TP = findings in both, FP = only in the generated report, FN = only in the reference;
                         precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = harmonic mean
                         (1.0 when neither report mentions anything abnormal and both say "No Finding")
  * labels_agree       : how many of the 14 observations get the same label (present / absent) in both

Prints a table and writes <input name>_scores.csv next to the input (or --out_csv).
"""
import argparse
import json
import os
import re

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from clinical_labeler import OBSERVATIONS, label_report
from evaluate import meteor, rouge_l, tokenize

_BLOCK = re.compile(r"###\s*(\S+)\s*\nREFERENCE:\s*(.*?)\s*\nGENERATED:\s*(.*?)(?=\n\s*\n###|\n###|\Z)", re.S)


# --------------------------------------------------------------------------- #
# reading the reports
# --------------------------------------------------------------------------- #
def parse_samples_text(text):
    rows = []
    for sid, ref, gen in _BLOCK.findall(text):
        gen = gen.strip()
        if gen:
            rows.append({"study_id": sid, "reference": ref.strip(), "generated": gen})
    return rows


def read_notebook(path):
    """Collect every '### <study> / REFERENCE / GENERATED' block printed in a notebook's outputs.
    If several cells print samples, the last one (latest epoch) wins for each study."""
    nb = json.load(open(path, encoding="utf-8"))
    found = {}
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain") or ""
            text = "".join(text) if isinstance(text, list) else text
            if "REFERENCE:" in text and "GENERATED:" in text:
                for row in parse_samples_text(text):
                    found[row["study_id"]] = row
    return list(found.values())


def read_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        return df[["study_id", "reference", "generated"]].fillna("").astype(str).to_dict("records")
    if ext == ".ipynb":
        return read_notebook(path)
    return parse_samples_text(open(path, encoding="utf-8").read())


# --------------------------------------------------------------------------- #
# scoring one report
# --------------------------------------------------------------------------- #
def score_report(ref, gen):
    ref_tok, gen_tok = tokenize(ref), tokenize(gen)
    sm = SmoothingFunction().method1          # same smoothing as evaluate.py
    ref_lab, gen_lab = label_report(ref), label_report(gen)
    ref_pos = {o for o in OBSERVATIONS if ref_lab[o]}
    gen_pos = {o for o in OBSERVATIONS if gen_lab[o]}
    tp, fp, fn = ref_pos & gen_pos, gen_pos - ref_pos, ref_pos - gen_pos
    p = len(tp) / len(gen_pos) if gen_pos else 0.0
    r = len(tp) / len(ref_pos) if ref_pos else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "BLEU-1": sentence_bleu([ref_tok], gen_tok, weights=(1, 0, 0, 0), smoothing_function=sm),
        "BLEU-4": sentence_bleu([ref_tok], gen_tok, smoothing_function=sm),
        "METEOR": meteor([ref], [gen]),
        "ROUGE-L": rouge_l([ref], [gen]),
        "clin_precision": p, "clin_recall": r, "clin_F1": f1,
        "labels_agree": sum(ref_lab[o] == gen_lab[o] for o in OBSERVATIONS),
        "findings_reference": "; ".join(sorted(ref_pos)),
        "found (TP)": "; ".join(sorted(tp)),
        "missed (FN)": "; ".join(sorted(fn)),
        "false (FP)": "; ".join(sorted(fp)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="samples_*.txt, a Kaggle .ipynb, or predictions_*.csv")
    ap.add_argument("--studies", nargs="+", default=None, help="only these study ids (e.g. s56108500)")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    rows = read_input(args.input)
    if args.studies:
        keep = {s.strip() for s in args.studies}
        missing = keep - {str(r["study_id"]) for r in rows}
        if missing:
            print(f"[warn] not found in {args.input}: {', '.join(sorted(missing))}")
        rows = [r for r in rows if str(r["study_id"]) in keep]
    if not rows:
        raise SystemExit(f"No reference/generated report pairs found in {args.input}")

    results = []
    for r in rows:
        res = {"study_id": r["study_id"]}
        res.update(score_report(r["reference"], r["generated"]))
        res["reference"], res["generated"] = r["reference"], r["generated"]
        results.append(res)
    df = pd.DataFrame(results)

    print(f"Scored {len(df)} report(s) from {args.input}\n")
    for _, x in df.iterrows():
        meteor_txt = f"{x['METEOR']:.3f}" if x["METEOR"] is not None else "n/a"
        print(f"=== {x['study_id']}")
        print(f"  BLEU-1 {x['BLEU-1']:.3f}  BLEU-4 {x['BLEU-4']:.3f}  METEOR {meteor_txt}  ROUGE-L {x['ROUGE-L']:.3f}")
        print(f"  clinical  P {x['clin_precision']:.2f}  R {x['clin_recall']:.2f}  F1 {x['clin_F1']:.2f}"
              f"  |  labels agree {x['labels_agree']}/14")
        print(f"  reference findings : {x['findings_reference'] or '-'}")
        print(f"  found (TP)         : {x['found (TP)'] or '-'}")
        print(f"  missed (FN)        : {x['missed (FN)'] or '-'}")
        print(f"  false (FP)         : {x['false (FP)'] or '-'}\n")

    out = args.out_csv or os.path.splitext(os.path.abspath(args.input))[0] + "_scores.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
