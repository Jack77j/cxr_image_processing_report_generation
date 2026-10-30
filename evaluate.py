"""
Step 5 - Score generated reports.

    python evaluate.py --predictions outputs/results/<run>/predictions_test.csv
    python evaluate.py --predictions ... --chexbert --radgraph     # optional model-based metrics

Language metrics (text overlap)      : BLEU-1..4 (corpus), METEOR, ROUGE-L, CIDEr-D
Clinical metrics (findings correct?)  : rule-based CheXpert labels on reference vs generated
                                        -> per-observation precision / recall / F1,
                                           micro & macro F1 over all 14 and over the 6 key findings
Optional (if installed)               : CheXbert F1 (`pip install f1chexbert`),
                                        RadGraph F1 (`pip install radgraph`)

Writes metrics.json and clinical_per_class.csv next to the predictions file.
"""
import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from clinical_labeler import KEY_OBSERVATIONS, OBSERVATIONS, label_reports

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[.,;:]")


def tokenize(text):
    return _TOKEN.findall(str(text).lower())


# --------------------------------------------------------------------------- #
# language metrics
# --------------------------------------------------------------------------- #
def bleu_scores(refs, hyps):
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    r = [[tokenize(x)] for x in refs]
    h = [tokenize(x) for x in hyps]
    sm = SmoothingFunction().method1
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25, 0.25, 0.25, 0.25)]
    return {f"BLEU-{i + 1}": corpus_bleu(r, h, weights=w, smoothing_function=sm) for i, w in enumerate(weights)}


def meteor(refs, hyps):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        return float(np.mean([meteor_score([tokenize(r)], tokenize(h)) for r, h in zip(refs, hyps)]))
    except Exception as e:  # wordnet not downloadable offline, etc.
        print(f"[warn] METEOR skipped ({type(e).__name__}). It needs NLTK WordNet: "
              f"python -m nltk.downloader wordnet omw-1.4")
        return None


def rouge_l(refs, hyps):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]))


def cider_d(refs, hyps, n=4, sigma=6.0):
    """CIDEr-D (Vedantam et al. 2015) with one reference per sample, as in coco-caption."""
    def ngrams(tokens, k):
        return Counter(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))

    ref_tok = [tokenize(r) for r in refs]
    hyp_tok = [tokenize(h) for h in hyps]
    N = len(refs)
    df = [Counter() for _ in range(n)]
    for toks in ref_tok:
        for k in range(n):
            for g in ngrams(toks, k + 1):
                df[k][g] += 1
    log_n = math.log(float(N))

    def vec(toks):
        vs, norms = [], []
        for k in range(n):
            counts = ngrams(toks, k + 1)
            v = {g: tf * (log_n - math.log(max(1.0, df[k][g]))) for g, tf in counts.items()}
            vs.append(v)
            norms.append(math.sqrt(sum(x * x for x in v.values())))
        return vs, norms, len(toks)

    scores = []
    for r, h in zip(ref_tok, hyp_tok):
        vr, nr, lr = vec(r)
        vh, nh, lh = vec(h)
        s = 0.0
        for k in range(n):
            dot = sum(min(vh[k][g], vr[k][g]) * vr[k][g] for g in vh[k] if g in vr[k])
            if nh[k] and nr[k]:
                s += dot / (nh[k] * nr[k]) * math.exp(-((lh - lr) ** 2) / (2 * sigma ** 2))
        scores.append(s / n * 10.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# clinical metrics
# --------------------------------------------------------------------------- #
def clinical_scores(refs, hyps):
    R = pd.DataFrame(label_reports(refs))[OBSERVATIONS]
    H = pd.DataFrame(label_reports(hyps))[OBSERVATIONS]
    rows = []
    for obs in OBSERVATIONS:
        tp = int(((R[obs] == 1) & (H[obs] == 1)).sum())
        fp = int(((R[obs] == 0) & (H[obs] == 1)).sum())
        fn = int(((R[obs] == 1) & (H[obs] == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"observation": obs, "support": int(R[obs].sum()), "predicted": int(H[obs].sum()),
                     "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn})
    per_class = pd.DataFrame(rows)

    def summary(sub):
        d = per_class[per_class.observation.isin(sub)]
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        mp = tp / (tp + fp) if tp + fp else 0.0
        mr = tp / (tp + fn) if tp + fn else 0.0
        return {"micro_precision": mp, "micro_recall": mr,
                "micro_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
                "macro_f1": float(d.f1.mean())}

    return per_class, {"all_14": summary(OBSERVATIONS), "key_6": summary(KEY_OBSERVATIONS)}


def optional_chexbert(refs, hyps):
    try:
        from f1chexbert import F1CheXbert
        import torch
        scorer = F1CheXbert(device="cuda" if torch.cuda.is_available() else "cpu")
        accuracy, _, report_14, report_5 = scorer(hyps=list(hyps), refs=list(refs))
        return {"accuracy": accuracy,
                "micro_f1_14": report_14["micro avg"]["f1-score"],
                "macro_f1_14": report_14["macro avg"]["f1-score"],
                "micro_f1_5": report_5["micro avg"]["f1-score"],
                "macro_f1_5": report_5["macro avg"]["f1-score"]}
    except Exception as e:
        print(f"[warn] CheXbert skipped: {e}")
        return None


def optional_radgraph(refs, hyps):
    try:
        from radgraph import F1RadGraph
        scorer = F1RadGraph(reward_level="partial")
        result = scorer(hyps=list(hyps), refs=list(refs))
        return {"radgraph_partial_f1": float(result[0] if isinstance(result, tuple) else result)}
    except Exception as e:
        print(f"[warn] RadGraph skipped: {e}")
        return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ref_column", default="reference")
    ap.add_argument("--pred_column", default="generated")
    ap.add_argument("--chexbert", action="store_true")
    ap.add_argument("--radgraph", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    refs = df[args.ref_column].fillna("").astype(str).tolist()
    hyps = df[args.pred_column].fillna("").astype(str).tolist()
    print(f"Evaluating {len(df):,} studies from {args.predictions}")

    metrics = {"n_studies": len(df)}
    metrics.update(bleu_scores(refs, hyps))
    metrics["METEOR"] = meteor(refs, hyps)
    metrics["ROUGE-L"] = rouge_l(refs, hyps)
    metrics["CIDEr-D"] = cider_d(refs, hyps)
    per_class, clinical = clinical_scores(refs, hyps)
    metrics["clinical_rule_based"] = clinical
    if args.chexbert:
        metrics["chexbert"] = optional_chexbert(refs, hyps)
    if args.radgraph:
        metrics["radgraph"] = optional_radgraph(refs, hyps)
    metrics["avg_words"] = {"reference": float(np.mean([len(r.split()) for r in refs])),
                            "generated": float(np.mean([len(h.split()) for h in hyps]))}
    # diversity check: a model that writes the same "normal" report for everyone has few unique reports
    abnormal = [o for o in KEY_OBSERVATIONS if o in per_class.observation.values]
    metrics["diversity"] = {
        "unique_generated_reports": int(len(set(hyps))),
        "pct_generated_with_any_key_finding": float(
            100 * pd.DataFrame(label_reports(hyps))[abnormal].any(axis=1).mean()),
        "pct_reference_with_any_key_finding": float(
            100 * pd.DataFrame(label_reports(refs))[abnormal].any(axis=1).mean()),
    }

    out_dir = os.path.dirname(os.path.abspath(args.predictions))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_class.to_csv(os.path.join(out_dir, "clinical_per_class.csv"), index=False)

    print("\n=== Language metrics ===")
    for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"]:
        v = metrics[k]
        print(f"  {k:8s} {v:.4f}" if v is not None else f"  {k:8s} n/a")
    print("\n=== Clinical (rule-based CheXpert labels) ===")
    for name, s in clinical.items():
        print(f"  {name:7s} micro-P {s['micro_precision']:.3f}  micro-R {s['micro_recall']:.3f}  "
              f"micro-F1 {s['micro_f1']:.3f}  macro-F1 {s['macro_f1']:.3f}")
    print(per_class[["observation", "support", "predicted", "precision", "recall", "f1"]]
          .round(3).to_string(index=False))
    for key in ("chexbert", "radgraph"):
        if metrics.get(key):
            print(f"\n=== {key} ===\n  " + json.dumps(metrics[key]))
    d = metrics["diversity"]
    print(f"\n=== Diversity ===\n  unique generated reports: {d['unique_generated_reports']} / {len(hyps)}"
          f"\n  studies with any of the 6 key findings: generated {d['pct_generated_with_any_key_finding']:.1f}%"
          f" | reference {d['pct_reference_with_any_key_finding']:.1f}%"
          f"\n  average length: generated {metrics['avg_words']['generated']:.1f} words"
          f" | reference {metrics['avg_words']['reference']:.1f} words")
    print(f"\nSaved metrics.json and clinical_per_class.csv in {out_dir}")


if __name__ == "__main__":
    main()
