"""
Step 1c - Image pre-processing: light denoising + CLAHE contrast enhancement.

    python preprocess_images.py --csv outputs/processed/train_subset_25k.csv outputs/processed/val.csv outputs/processed/test.csv

Every image the model will see (training subset, validation and test) is processed ONCE, with the
same settings, and written to config.PROC_IMAGE_ROOT with the same relative path as the original:

    <IMAGE_ROOT>/files/p10/p10000032/s50414267/xxx.jpg  ->  <PROC_IMAGE_ROOT>/files/p10/p10000032/s50414267/xxx.jpg

train.py / generate.py then simply get  --image_root <PROC_IMAGE_ROOT>. Nothing else changes: the CSVs,
the model, the on-the-fly augmentation and the BLIP-2 image processor (resize 224 x 224 + normalise)
are exactly as in the original pipeline. The original dataset images are only read, never modified.

Why val and test are processed too: the model must see the same kind of image at test time as in
training; otherwise the test images would look different (less contrast, more noise) and the
comparison with the original run would be unfair.

Per image (grayscale, 8-bit):
  1. downscale so the longest side is at most PREPROC_MAX_SIDE (default 512, keeps the aspect ratio;
     the model input is 224 x 224 anyway, so no detail the model could use is lost)
  2. light denoising    : median filter 3 x 3            (PREPROC_DENOISE = "median")
                          or bilateral / non-local means  ("bilateral" / "nlmeans" / "none")
  3. CLAHE              : clip limit 2.0, 8 x 8 tiles     (PREPROC_CLAHE_CLIP / PREPROC_CLAHE_TILES)
     Denoising runs BEFORE CLAHE, because CLAHE would otherwise amplify the noise.
  4. saved as JPEG (quality 95)

Only the images that are actually used are processed: the first --max_images path(s) of every study
(1 = the frontal image, as in the 25k run). Already processed files are skipped, so the step can be
re-run or resumed. Use --overwrite after changing the settings.

Writes  <PROC_IMAGE_ROOT>/preprocessing_summary.json   (settings + counts; also the "done" marker)
        outputs/figures/preprocessing_examples.png       (original | denoised | denoised + CLAHE)
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# the image pipeline
# --------------------------------------------------------------------------- #
def resize_max_side(img, max_side):
    if not max_side:
        return img
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def denoise(img, method, median_ksize=3):
    if method == "median":
        return cv2.medianBlur(img, median_ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    if method == "nlmeans":
        return cv2.fastNlMeansDenoising(img, None, h=5, templateWindowSize=7, searchWindowSize=21)
    if method in ("none", None, ""):
        return img
    raise ValueError(f"unknown denoise method: {method}")


def apply_clahe(img, clip, tiles):
    if not clip:
        return img
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


def preprocess_array(gray, settings):
    """gray: 2-D uint8 array -> processed 2-D uint8 array (used by this script and by generate.py)."""
    img = resize_max_side(gray, settings["max_side"])
    img = denoise(img, settings["denoise"], settings["median_ksize"])
    return apply_clahe(img, settings["clahe_clip"], settings["clahe_tiles"])


def default_settings():
    return {"max_side": config.PREPROC_MAX_SIDE, "denoise": config.PREPROC_DENOISE,
            "median_ksize": config.PREPROC_MEDIAN_KSIZE, "clahe_clip": config.PREPROC_CLAHE_CLIP,
            "clahe_tiles": config.PREPROC_CLAHE_TILES, "jpeg_quality": config.PREPROC_JPEG_QUALITY}


def preprocess_file(src_path, settings):
    """Read any image file and return the processed grayscale array (None if unreadable)."""
    gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    return None if gray is None else preprocess_array(gray, settings)


# --------------------------------------------------------------------------- #
# batch processing
# --------------------------------------------------------------------------- #
_S = {}


def _init(settings, src_root, dst_root, overwrite):
    _S.update(settings=settings, src_root=src_root, dst_root=dst_root, overwrite=overwrite)
    cv2.setNumThreads(1)


def _work(rel):
    dst = os.path.join(_S["dst_root"], rel)
    if not _S["overwrite"] and os.path.isfile(dst):
        return rel, "skipped"
    out = preprocess_file(os.path.join(_S["src_root"], rel), _S["settings"])
    if out is None:
        return rel, "unreadable"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, _S["settings"]["jpeg_quality"]])
    return rel, "done" if ok else "write_failed"


def collect_paths(csv_paths, max_images):
    rels, per_csv = [], {}
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["image_paths"])
        n = 0
        for s in df["image_paths"]:
            for rel in json.loads(s)[:max_images]:
                rels.append(rel)
                n += 1
        per_csv[os.path.basename(p)] = {"studies": len(df), "images": n}
    return list(dict.fromkeys(rels)), per_csv          # unique, order kept


def example_figure(rels, src_root, settings, out_png, n=4, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    pick = [rels[i] for i in rng.choice(len(rels), size=min(n, len(rels)), replace=False)]
    fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3.1 * len(pick)))
    axes = np.atleast_2d(axes)
    titles = ["original", f"denoised ({settings['denoise']})", f"denoised + CLAHE (clip {settings['clahe_clip']})"]
    for r, rel in enumerate(pick):
        gray = cv2.imread(os.path.join(src_root, rel), cv2.IMREAD_GRAYSCALE)
        small = resize_max_side(gray, settings["max_side"])
        den = denoise(small, settings["denoise"], settings["median_ksize"])
        fin = apply_clahe(den, settings["clahe_clip"], settings["clahe_tiles"])
        for c, im in enumerate((small, den, fin)):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(rel.split("/")[-2], fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="study CSVs whose images are processed")
    ap.add_argument("--image_root", default=config.IMAGE_ROOT, help="original images (read only)")
    ap.add_argument("--out_root", default=config.PROC_IMAGE_ROOT, help="where processed images are written")
    ap.add_argument("--max_images", type=int, default=1, help="images per study to process (frontal first)")
    ap.add_argument("--max_side", type=int, default=config.PREPROC_MAX_SIDE)
    ap.add_argument("--denoise", default=config.PREPROC_DENOISE, choices=["median", "bilateral", "nlmeans", "none"])
    ap.add_argument("--median_ksize", type=int, default=config.PREPROC_MEDIAN_KSIZE)
    ap.add_argument("--clahe_clip", type=float, default=config.PREPROC_CLAHE_CLIP, help="0 = no CLAHE")
    ap.add_argument("--clahe_tiles", type=int, default=config.PREPROC_CLAHE_TILES)
    ap.add_argument("--jpeg_quality", type=int, default=config.PREPROC_JPEG_QUALITY)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--overwrite", action="store_true", help="re-process files that already exist")
    args = ap.parse_args()

    settings = {"max_side": args.max_side, "denoise": args.denoise, "median_ksize": args.median_ksize,
                "clahe_clip": args.clahe_clip, "clahe_tiles": args.clahe_tiles, "jpeg_quality": args.jpeg_quality}
    summary_path = os.path.join(args.out_root, "preprocessing_summary.json")
    if os.path.isfile(summary_path) and not args.overwrite:
        old = json.load(open(summary_path)).get("settings", {})
        if old and old != settings:
            raise SystemExit(f"{args.out_root} was made with different settings {old}.\n"
                             f"Use --overwrite (or another --out_root) to re-process with {settings}.")

    rels, per_csv = collect_paths(args.csv, args.max_images)
    print(f"{len(rels):,} unique images to process from {per_csv}")
    print(f"settings: {settings}\n{args.image_root}  ->  {args.out_root}")
    os.makedirs(args.out_root, exist_ok=True)

    counts, problems, t0 = {}, [], time.time()
    with Pool(args.workers, initializer=_init, initargs=(settings, args.image_root, args.out_root, args.overwrite)) as pool:
        for i, (rel, status) in enumerate(pool.imap_unordered(_work, rels, chunksize=64), 1):
            counts[status] = counts.get(status, 0) + 1
            if status in ("unreadable", "write_failed"):
                problems.append(rel)
            if i % 2000 == 0 or i == len(rels):
                el = time.time() - t0
                print(f"  {i:,}/{len(rels):,}  {counts}  {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(rels) - i) / 60:.1f} min left", flush=True)

    missing = [r for r in rels if not os.path.isfile(os.path.join(args.out_root, r))]
    summary = {"settings": settings, "image_root": args.image_root, "out_root": args.out_root,
               "max_images_per_study": args.max_images, "csvs": per_csv, "unique_images": len(rels),
               "status_counts": counts, "missing_after_run": len(missing), "problem_files": problems[:50],
               "minutes": round((time.time() - t0) / 60, 1)}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("status_counts", "missing_after_run", "minutes")}, indent=2))
    print(f"Saved {summary_path}")

    try:
        example_figure(rels, args.image_root, settings, os.path.join(config.FIGURES_DIR, "preprocessing_examples.png"))
    except Exception as e:  # the figure is optional
        print(f"[warn] example figure skipped: {e}")

    if missing:
        raise SystemExit(f"{len(missing)} images could not be processed, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
