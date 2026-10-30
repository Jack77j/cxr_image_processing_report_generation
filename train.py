"""
Step 3 - Fine-tune the study-level multi-image BLIP-2.

    python train.py                                   # full run with config.py defaults
    python train.py --max_train_studies 20000 --epochs 2      # smaller first experiment
    python train.py --fusion concat --run_name concat_v1      # ablation
    python train.py --resume                          # continue an interrupted run
    python train.py --resume_from /kaggle/input/<previous-output>/outputs/checkpoints/<run>
                                                      # continue in a NEW Kaggle session

Outputs in outputs/checkpoints/<run_name>/ :
    run_config.json   - every setting needed to rebuild the model (used by generate.py)
    best.pt           - trainable weights with the lowest validation loss
    last.pt           - latest weights + optimizer/scheduler state (for --resume)
    train_log.csv     - loss per logging step and per epoch
    samples_epoch*.txt- a few generated vs reference reports after each epoch
    STATUS.txt        - "complete" or "incomplete (resume next session)"

--max_hours makes training stop cleanly before a time limit (Kaggle: 12 h per session):
it validates, updates best.pt if improved, saves last.pt and marks the run incomplete.
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

import config
from dataset import StudyCXRDataset, collate_fn, load_split
from model import (autocast_ctx, build_model, load_trainable_state, resolve_precision,
                   trainable_state_dict)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_csv", default=os.path.join(config.PROCESSED_DIR, "train.csv"))
    ap.add_argument("--val_csv", default=os.path.join(config.PROCESSED_DIR, "val.csv"))
    ap.add_argument("--image_root", default=config.PROC_IMAGE_ROOT,
                    help="pre-processed images from preprocess_images.py (original images: config.IMAGE_ROOT)")
    ap.add_argument("--checkpoint_root", default=config.CHECKPOINT_DIR)
    ap.add_argument("--run_name", default=None, help="sub-folder name (default: <fusion>_<timestamp>)")
    # model
    ap.add_argument("--model_name", default=config.MODEL_NAME)
    ap.add_argument("--fusion", default=config.FUSION, choices=["mean", "concat"])
    ap.add_argument("--max_images", type=int, default=config.MAX_IMAGES_PER_STUDY)
    ap.add_argument("--quantization", default=config.LM_QUANTIZATION, choices=["none", "8bit", "4bit"])
    ap.add_argument("--use_lora", action="store_true", default=config.USE_LORA)
    ap.add_argument("--lora_r", type=int, default=config.LORA_R)
    ap.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=config.LORA_DROPOUT)
    ap.add_argument("--max_target_tokens", type=int, default=config.MAX_TARGET_TOKENS)
    ap.add_argument("--precision", default=config.PRECISION, choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false",
                    default=config.GRADIENT_CHECKPOINTING)
    # training
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--grad_accum_steps", type=int, default=config.GRAD_ACCUM_STEPS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--warmup_ratio", type=float, default=config.WARMUP_RATIO)
    ap.add_argument("--text_augment_prob", type=float, default=config.AUGMENT_TEXT_PROB)
    ap.add_argument("--no_image_augment", dest="image_augment", action="store_false", default=config.IMAGE_AUGMENT)
    ap.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--max_train_studies", type=int, default=None, help="random subset for quick runs")
    ap.add_argument("--max_val_studies", type=int, default=config.MAX_VAL_STUDIES)
    ap.add_argument("--log_every", type=int, default=config.LOG_EVERY)
    ap.add_argument("--save_every_steps", type=int, default=config.SAVE_EVERY_STEPS,
                    help="also write last.pt every N optimizer steps (long epochs)")
    ap.add_argument("--max_hours", type=float, default=config.MAX_HOURS,
                    help="stop cleanly after this many hours (Kaggle sessions end at 12 h)")
    ap.add_argument("--num_samples", type=int, default=4, help="reports to generate after each epoch")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    ap.add_argument("--resume_from", default=None,
                    help="a run folder from a previous session (e.g. under /kaggle/input); it is copied "
                         "into --checkpoint_root and training resumes from its last.pt")
    ap.add_argument("--seed", type=int, default=config.SEED)
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def find_resume_run(root):
    runs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "last.pt"))]
    if not runs:
        raise FileNotFoundError(f"No run with last.pt found under {root}")
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "last.pt")))


@torch.no_grad()
def validation_loss(model, loader, device, dtype):
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = to_device(batch, device)
        with autocast_ctx(device, dtype):
            out = model(batch["pixel_values"], batch["image_mask"], batch["input_ids"],
                        batch["attention_mask"], batch["labels"])
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def write_samples(model, dataset, tokenizer, device, dtype, path, n):
    if n <= 0 or len(dataset) == 0:
        return
    model.eval()
    idx = list(range(min(n, len(dataset))))
    batch = to_device(collate_fn([dataset[i] for i in idx]), device)
    with autocast_ctx(device, dtype):
        preds = model.generate(batch["pixel_values"], batch["image_mask"], tokenizer,
                               max_new_tokens=config.MAX_NEW_TOKENS, num_beams=config.NUM_BEAMS,
                               repetition_penalty=config.REPETITION_PENALTY)
    with open(path, "w", encoding="utf-8") as f:
        for sid, ref, pred in zip(batch["study_id"], batch["reference"], preds):
            f.write(f"### {sid}\nREFERENCE: {ref}\nGENERATED: {pred}\n\n")
    print(f"  sample reports -> {path}\n  e.g. GENERATED: {preds[0][:200]}")
    model.train()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] No CUDA GPU found - training BLIP-2 on CPU is only useful for debugging.")

    # ---------------- run folder / resume ----------------
    state = None
    if args.resume_from:
        src = os.path.normpath(args.resume_from)
        if not os.path.isfile(os.path.join(src, "last.pt")):
            raise FileNotFoundError(f"No last.pt in {src}")
        args.run_name = os.path.basename(src)
        dst = os.path.join(args.checkpoint_root, args.run_name)
        if os.path.abspath(src) != os.path.abspath(dst):
            print(f"Copying previous session's run {src} -> {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        args.resume = True
    if args.resume:
        run_dir = (os.path.join(args.checkpoint_root, args.run_name) if args.run_name
                   else find_resume_run(args.checkpoint_root))
        with open(os.path.join(run_dir, "run_config.json")) as f:
            saved = json.load(f)
        # restore the original model/training settings, but keep this session's paths & runtime options
        keep = {"resume", "resume_from", "train_csv", "val_csv", "image_root", "checkpoint_root",
                "num_workers", "max_hours", "save_every_steps", "log_every", "num_samples"}
        for k, v in saved["train_args"].items():
            if k not in keep:
                setattr(args, k, v)
        # the run must continue on the SAME training file (e.g. train_subset_25k.csv): if this session
        # did not name one explicitly, look for the original file name in this session's processed dir
        for key in ("train_csv", "val_csv"):
            old = saved["train_args"].get(key)
            default = os.path.join(config.PROCESSED_DIR, "train.csv" if key == "train_csv" else "val.csv")
            if old and getattr(args, key) == default and os.path.basename(old) != os.path.basename(default):
                cand = os.path.join(config.PROCESSED_DIR, os.path.basename(old))
                if not os.path.isfile(cand):
                    raise FileNotFoundError(f"This run was trained on {os.path.basename(old)}, which is not in "
                                            f"{config.PROCESSED_DIR}. Recreate it or pass --{key}.")
                setattr(args, key, cand)
                print(f"Resuming with the run's original {key}: {cand}")
        state = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)
        print(f"Resuming {run_dir} from epoch {state['epoch'] + 1}, step {state['step_in_epoch']}")
    else:
        run_name = args.run_name or f"{args.fusion}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(args.checkpoint_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        args.run_name = run_name

    # ---------------- model ----------------
    build = dict(model_name=args.model_name, fusion=args.fusion, max_images=args.max_images,
                 quantization=args.quantization, use_lora=args.use_lora, lora_r=args.lora_r,
                 lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, precision=args.precision)
    model, processor, info = build_model(**build, gradient_checkpointing=args.gradient_checkpointing,
                                         device=device)
    compute_dtype, _ = resolve_precision(args.precision, device)

    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({"build": build, "train_args": vars(args), "model_info": info,
                   "max_target_tokens": args.max_target_tokens}, f, indent=2)

    # ---------------- data ----------------
    train_df = load_split(args.train_csv, args.max_train_studies, args.seed)
    val_df = load_split(args.val_csv, args.max_val_studies, args.seed)
    train_ds = StudyCXRDataset(train_df, args.image_root, processor, args.max_images, args.max_target_tokens,
                               is_train=True, text_augment_prob=args.text_augment_prob,
                               image_augment=args.image_augment)
    val_ds = StudyCXRDataset(val_df, args.image_root, processor, args.max_images, args.max_target_tokens)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn,
                     pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"Train studies: {len(train_ds):,} | Val studies: {len(val_ds):,}")

    # ---------------- optimisation ----------------
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if (p.ndim < 2 or "norm" in n.lower() or n.endswith("bias")) else decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=info["use_grad_scaler"])

    start_epoch, start_batch, best_val, global_step = 0, 0, float("inf"), 0
    if state is not None:
        load_trainable_state(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch = state["epoch"], state["step_in_epoch"]
        best_val, global_step = state["best_val"], state["global_step"]

    log_path = os.path.join(run_dir, "train_log.csv")
    new_log = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if new_log:
        logger.writerow(["time", "epoch", "global_step", "train_loss", "val_loss", "lr"])

    def save_last(epoch, step_in_epoch):
        torch.save({"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "step_in_epoch": step_in_epoch, "best_val": best_val,
                    "global_step": global_step}, os.path.join(run_dir, "last.pt"))

    def set_status(text):
        with open(os.path.join(run_dir, "STATUS.txt"), "w") as f:
            f.write(text + "\n")

    # ---------------- training loop ----------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    t_start = time.time()
    out_of_time = False
    set_status("incomplete (training in progress)")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        # deterministic shuffle per epoch so an interrupted epoch can be resumed exactly
        perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed + epoch))
        skip = start_batch * args.batch_size if epoch == start_epoch else 0
        loader = DataLoader(Subset(train_ds, perm[skip:].tolist()), shuffle=False, drop_last=False, **loader_kw)

        running, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(desc=f"Epoch {epoch + 1}/{args.epochs}",
                   initial=skip // args.batch_size, total=batches_per_epoch)
        for i, batch in enumerate(loader):
            batch_idx = skip // args.batch_size + i
            batch = to_device(batch, device)
            with autocast_ctx(device, compute_dtype):
                out = model(batch["pixel_values"], batch["image_mask"], batch["input_ids"],
                            batch["attention_mask"], batch["labels"])
                loss = out.loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += out.loss.item()
            count += 1

            last_batch = batch_idx + 1 == batches_per_epoch
            if (batch_idx + 1) % args.grad_accum_steps == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    avg = running / max(count, 1)
                    logger.writerow([time.strftime("%H:%M:%S"), epoch + 1, global_step, f"{avg:.4f}", "",
                                     f"{scheduler.get_last_lr()[0]:.2e}"])
                    log_file.flush()
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    save_last(epoch, batch_idx + 1)
                if args.max_hours and time.time() - t_start > args.max_hours * 3600 and not last_batch:
                    out_of_time = True
                    stop_batch = batch_idx + 1
                    break
            bar.update(1)
            bar.set_postfix(loss=f"{running / max(count, 1):.4f}")
        bar.close()

        if out_of_time:
            print(f"Reached --max_hours={args.max_hours}; stopping at epoch {epoch + 1}, "
                  f"batch {stop_batch}/{batches_per_epoch}.")
            val_loss = validation_loss(model, val_loader, device, compute_dtype)
            print(f"  val_loss={val_loss:.4f}")
            logger.writerow([time.strftime("%H:%M:%S"), epoch + 1, global_step,
                             f"{running / max(count, 1):.4f}", f"{val_loss:.4f}",
                             f"{scheduler.get_last_lr()[0]:.2e}"])
            log_file.flush()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": trainable_state_dict(model), "epoch": epoch + 1, "val_loss": val_loss,
                            "partial_epoch": True}, os.path.join(run_dir, "best.pt"))
                print(f"  new best model saved (val_loss={val_loss:.4f})")
            save_last(epoch, stop_batch)
            write_samples(model, val_ds, processor.tokenizer, device, compute_dtype,
                          os.path.join(run_dir, f"samples_step{global_step}.txt"), args.num_samples)
            set_status(f"incomplete (resume next session): epoch {epoch + 1}, "
                       f"batch {stop_batch}/{batches_per_epoch}, global_step {global_step}")
            log_file.close()
            print(f"Saved last.pt in {run_dir}. Continue next session with --resume_from <that folder>")
            return

        train_loss = running / max(count, 1)
        val_loss = validation_loss(model, val_loader, device, compute_dtype)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        logger.writerow([time.strftime("%H:%M:%S"), epoch + 1, global_step, f"{train_loss:.4f}",
                         f"{val_loss:.4f}", f"{scheduler.get_last_lr()[0]:.2e}"])
        log_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": trainable_state_dict(model), "epoch": epoch + 1, "val_loss": val_loss},
                       os.path.join(run_dir, "best.pt"))
            print(f"  new best model saved (val_loss={val_loss:.4f})")
        save_last(epoch + 1, 0)
        write_samples(model, val_ds, processor.tokenizer, device, compute_dtype,
                      os.path.join(run_dir, f"samples_epoch{epoch + 1}.txt"), args.num_samples)

    log_file.close()
    set_status(f"complete: {args.epochs} epochs, best val_loss {best_val:.4f}")
    print(f"Done. Best val_loss={best_val:.4f}. Checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
