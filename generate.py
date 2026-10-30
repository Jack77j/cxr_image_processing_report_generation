"""
Step 4 - Generate reports with a trained checkpoint.

Batch mode (whole split -> predictions CSV, used by evaluate.py):
    python generate.py --run_dir outputs/checkpoints/<run_name>
    python generate.py --run_dir ... --split_csv outputs/processed/test.csv --max_studies 500

Single study (a patient's current visit):
    python generate.py --run_dir ... --images frontal.jpg lateral.jpg
    python generate.py --run_dir ... --study_dir "D:/.../files/p10/p10003502/s51180958"

Images are ordered as given (put the frontal view first; that is what training used).
In single-study mode the raw images get the same denoising + CLAHE as the training images
(preprocess_images.py); batch mode reads the already processed copies from --image_root.
"""
import argparse
import glob
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from dataset import StudyCXRDataset, collate_fn, load_split
from model import autocast_ctx, load_for_inference, resolve_precision
from preprocess_images import default_settings, preprocess_file


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="checkpoint folder written by train.py")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--split_csv", default=os.path.join(config.PROCESSED_DIR, "test.csv"))
    ap.add_argument("--image_root", default=config.PROC_IMAGE_ROOT,
                    help="pre-processed images from preprocess_images.py")
    ap.add_argument("--out_csv", default=None, help="default: outputs/results/<run>/predictions_<split>.csv")
    ap.add_argument("--max_studies", type=int, default=None)
    ap.add_argument("--images", nargs="+", default=None, help="generate for these image files only")
    ap.add_argument("--study_dir", default=None, help="generate for all .jpg/.png in this folder")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    ap.add_argument("--num_beams", type=int, default=config.NUM_BEAMS)
    ap.add_argument("--repetition_penalty", type=float, default=config.REPETITION_PENALTY)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=config.NO_REPEAT_NGRAM_SIZE)
    ap.add_argument("--precision", default=None, help="override training precision (fp16/bf16/fp32)")
    return ap.parse_args()


def gen_kwargs(args):
    return dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size)


def single_study(args, model, processor, device, dtype, max_images):
    paths = args.images or sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                                  for p in glob.glob(os.path.join(args.study_dir, ext)))
    if not paths:
        raise FileNotFoundError("No images found.")
    if len(paths) > max_images:
        print(f"[info] using the first {max_images} of {len(paths)} images: {paths[:max_images]}")
    paths = paths[:max_images]
    # raw image files: apply the same denoising + CLAHE that the training images got
    imgs = []
    for p in paths:
        arr = preprocess_file(p, default_settings())
        if arr is None:
            raise FileNotFoundError(f"cannot read image: {p}")
        imgs.append(Image.fromarray(arr).convert("RGB"))
    pixels = processor.image_processor(images=imgs,
                                       return_tensors="pt")["pixel_values"]
    n = pixels.shape[0]
    if n < max_images:
        pixels = torch.cat([pixels, torch.zeros(max_images - n, *pixels.shape[1:])], 0)
    mask = torch.zeros(1, max_images, dtype=torch.long)
    mask[0, :n] = 1
    with autocast_ctx(device, dtype):
        report = model.generate(pixels.unsqueeze(0).to(device), mask.to(device), processor.tokenizer,
                                **gen_kwargs(args))[0]
    print("\nImages:", *paths, sep="\n  ")
    print("\nGENERATED FINDINGS:\n" + report)


def batch_mode(args, model, processor, device, dtype, max_images, max_tokens, run_name):
    df = load_split(args.split_csv, args.max_studies)
    ds = StudyCXRDataset(df, args.image_root, processor, max_images, max_tokens)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)
    rows = []
    for batch in tqdm(loader, desc="Generating"):
        with autocast_ctx(device, dtype):
            preds = model.generate(batch["pixel_values"].to(device), batch["image_mask"].to(device),
                                   processor.tokenizer, **gen_kwargs(args))
        for sub, sid, n, ref, pred in zip(batch["subject_id"], batch["study_id"],
                                          batch["image_mask"].sum(1).tolist(), batch["reference"], preds):
            rows.append({"subject_id": sub, "study_id": sid, "n_images_used": n,
                         "reference": ref, "generated": pred})
    split = os.path.splitext(os.path.basename(args.split_csv))[0]
    out = args.out_csv or os.path.join(config.RESULTS_DIR, run_name, f"predictions_{split}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows):,} predictions -> {out}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights
    if not os.path.isfile(os.path.join(args.run_dir, weights)) and os.path.isfile(os.path.join(args.run_dir, "last.pt")):
        print(f"[info] {weights} not found in {args.run_dir}; using last.pt")
        weights = "last.pt"
    model, processor, info, run_cfg = load_for_inference(args.run_dir, weights, device, args.precision)
    dtype, _ = resolve_precision(run_cfg["build"]["precision"], device)
    max_images = run_cfg["build"]["max_images"]
    max_tokens = run_cfg.get("max_target_tokens", config.MAX_TARGET_TOKENS)
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    if args.images or args.study_dir:
        single_study(args, model, processor, device, dtype, max_images)
    else:
        batch_mode(args, model, processor, device, dtype, max_images, max_tokens, run_name)


if __name__ == "__main__":
    main()
