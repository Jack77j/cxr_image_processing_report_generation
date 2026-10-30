"""
PyTorch Dataset for study-level report generation.

One sample = one study = (up to MAX_IMAGES_PER_STUDY X-rays, one cleaned target report).
Images are already ordered frontal-first by prepare_data.py, so taking the first N
gives "frontal + lateral" for typical studies.
"""
import json
import os
import random

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def load_split(csv_path, max_studies=None, seed=42):
    df = pd.read_csv(csv_path)
    df["image_paths"] = df["image_paths"].map(json.loads)
    df["views"] = df["views"].map(json.loads)
    df["target"] = df["target"].fillna("").astype(str)
    df["target_augment"] = df["target_augment"].fillna("").astype(str)
    if max_studies is not None and len(df) > max_studies:
        df = df.sample(n=max_studies, random_state=seed)
    return df.reset_index(drop=True)


def augment_xray(img: Image.Image) -> Image.Image:
    """Light, anatomy-preserving augmentation. No horizontal flips (laterality matters)."""
    w, h = img.size
    if random.random() < 0.5:                                   # small rotation
        img = img.rotate(random.uniform(-7, 7), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if random.random() < 0.5:                                   # random crop 90-100 %, resized back
        scale = random.uniform(0.90, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    if random.random() < 0.5:                                   # brightness / contrast jitter
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    return img


def encode_target(tokenizer, text, max_tokens):
    """
    Tokenise a report for a decoder-only LM:
        [BOS] report tokens ... [EOS] [PAD] [PAD] ...
    Labels ignore BOS and padding (-100) so the model learns to *stop* via EOS.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][: max_tokens - 2]
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    ids = [bos] + ids + [tokenizer.eos_token_id]
    n = len(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = ids + [pad_id] * (max_tokens - n)
    attention = [1] * n + [0] * (max_tokens - n)
    labels = [-100] + ids[1:] + [-100] * (max_tokens - n)
    return (torch.tensor(input_ids), torch.tensor(attention), torch.tensor(labels))


class StudyCXRDataset(Dataset):
    def __init__(self, df, image_root, processor, max_images, max_target_tokens,
                 is_train=False, text_augment_prob=0.0, image_augment=False):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.max_images = max_images
        self.max_target_tokens = max_target_tokens
        self.is_train = is_train
        self.text_augment_prob = text_augment_prob if is_train else 0.0
        self.image_augment = image_augment and is_train
        size = self.image_processor.size
        self.image_size = size.get("height", size.get("shortest_edge", 224))

    def __len__(self):
        return len(self.df)

    def _load(self, rel_path):
        try:
            img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        except (FileNotFoundError, OSError):
            print(f"[warn] unreadable image, using blank: {rel_path}")
            img = Image.new("RGB", (self.image_size, self.image_size))
        return augment_xray(img) if self.image_augment else img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        paths = row["image_paths"][: self.max_images]
        pixels = self.image_processor(images=[self._load(p) for p in paths],
                                      return_tensors="pt")["pixel_values"]      # (n, 3, H, W)
        n = pixels.shape[0]
        if n < self.max_images:
            pad = torch.zeros((self.max_images - n, *pixels.shape[1:]), dtype=pixels.dtype)
            pixels = torch.cat([pixels, pad], dim=0)
        image_mask = torch.zeros(self.max_images, dtype=torch.long)
        image_mask[:n] = 1

        target = row["target"]
        if self.text_augment_prob > 0 and random.random() < self.text_augment_prob and row["target_augment"]:
            target = row["target_augment"]
        input_ids, attention_mask, labels = encode_target(self.tokenizer, target, self.max_target_tokens)

        return {
            "study_id": row["study_id"],
            "subject_id": int(row["subject_id"]),
            "pixel_values": pixels,
            "image_mask": image_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "reference": row["target"],
        }


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return out
