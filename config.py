"""
Central configuration for the study-level CXR report-generation pipeline.

Edit the paths in the PATHS section once; every script reads its defaults from here.
Most values can also be overridden from the command line (see each script's --help).

On Kaggle (/kaggle/input exists) the paths are found automatically:
  * the two CSVs and the image folder are searched for under /kaggle/input
  * all outputs go to /kaggle/working/outputs (kept when you "Save Version")
  * batch size / workers / session time limit switch to T4-friendly values
"""
import os
import re

ON_KAGGLE = os.path.isdir("/kaggle/input")


def _bounded_dirs(root, max_depth=5):
    """Yield directories under root breadth-first, never descending into the image tree."""
    level = [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            yield d
            try:
                entries = [e for e in os.scandir(d) if e.is_dir()]
            except OSError:
                continue
            for e in entries:
                if e.name != "files" and not re.fullmatch(r"p\d+|s\d+", e.name):
                    nxt.append(e.path)
        level = nxt


def _find_kaggle_paths(root="/kaggle/input"):
    train_csv = val_csv = image_root = None
    for d in _bounded_dirs(root):
        if train_csv is None and os.path.isfile(os.path.join(d, "mimic_cxr_aug_train.csv")):
            train_csv = os.path.join(d, "mimic_cxr_aug_train.csv")
        if val_csv is None and os.path.isfile(os.path.join(d, "mimic_cxr_aug_validate.csv")):
            val_csv = os.path.join(d, "mimic_cxr_aug_validate.csv")
        if image_root is None and os.path.isdir(os.path.join(d, "files", "p10")):
            image_root = d
        if train_csv and val_csv and image_root:
            break
    missing = [n for n, v in (("mimic_cxr_aug_train.csv", train_csv), ("mimic_cxr_aug_validate.csv", val_csv),
                              ("image folder containing files/p10", image_root)) if v is None]
    if missing:
        print(f"[config] Could not find under {root}: {', '.join(missing)}. "
              f"Attach the dataset to the notebook (Add Input) or set the paths in config.py.")
    return train_csv, val_csv, image_root


# ================================ PATHS =====================================
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))

if ON_KAGGLE:
    RAW_TRAIN_CSV, RAW_VAL_CSV, IMAGE_ROOT = _find_kaggle_paths()
    OUTPUT_ROOT = "/kaggle/working/outputs"
else:
    DATA_ROOT = r"D:\BioMed\archive (5)"
    IMAGE_ROOT = os.path.join(DATA_ROOT, "official_data_iccv_final")     # folder that contains files/p10/...
    RAW_TRAIN_CSV = os.path.join(DATA_ROOT, "mimic_cxr_aug_train.csv")
    RAW_VAL_CSV = os.path.join(DATA_ROOT, "mimic_cxr_aug_validate.csv")   # used as the held-out TEST set
    OUTPUT_ROOT = os.path.join(PIPELINE_ROOT, "outputs")

PROCESSED_DIR = os.path.join(OUTPUT_ROOT, "processed")     # study-level CSVs from prepare_data.py
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")  # model weights from train.py
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")         # predictions + metrics
FIGURES_DIR = os.path.join(OUTPUT_ROOT, "figures")         # EDA plots

# ============================ DATA PREPARATION ==============================
# Which part of each report becomes the training target:
#   "findings_fallback_impression" : Findings; if Findings is empty use Impression   (default)
#   "findings_only"                : Findings only; studies with empty Findings are dropped
#   "findings_impression"          : Findings + Impression joined together
TARGET_SECTION = "findings_fallback_impression"

# The model sees ONE study, so sentences that refer to earlier scans
# ("unchanged from prior", "interval increase", "compared to ...") cannot be
# grounded in the input image. "drop" removes them, "keep" leaves the text as-is.
COMPARISON_MODE = "drop"

MIN_TARGET_WORDS = 3          # studies whose cleaned target is shorter than this are dropped
VAL_FRACTION = 0.05           # fraction of TRAIN patients held out for validation (split by subject_id)
CHECK_IMAGES_EXIST = True     # drop images that are missing on disk
SEED = 42

# ===================== TRAINING SUBSET (select_subset.py) ===================
# Optional smaller, finding-balanced training set with a fixed image budget.
SUBSET_IMAGES = None          # e.g. 25000; None = train on the full train.csv
SUBSET_IMAGES_PER_STUDY = 1   # 1 = one frontal image per study (more distinct reports per image budget)
                              # 2 = frontal + lateral pairs (half as many studies)
SUBSET_MIN_WORDS = 8          # drop one-line reports ("No acute process.")
SUBSET_MAX_WORDS = 120        # drop runaway reports (the 99th percentile is ~100 words)
SUBSET_MAX_PER_PATIENT = 2    # patient diversity
SUBSET_NORMAL_MAX_FRAC = 0.30 # at most 30 % of the subset may be normal ("No Finding") studies
SUBSET_BALANCE_ALPHA = 0.5    # 0 = plain random, 1 = fully inverse-frequency; 0.5 = sqrt re-weighting
SUBSET_IMPRESSION_WEIGHT = 0.5  # reports that fell back to the Impression are picked half as often

# ================= IMAGE PRE-PROCESSING (preprocess_images.py) =================
# Applied once, before training, to every image the model sees (training subset, val, test).
# Processed copies go to PROC_IMAGE_ROOT; the original dataset images are never modified.
PROC_IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images_proc")
PREPROC_MAX_SIDE = 512        # downscale so the longest side is <= this (model input is 224 anyway); None = keep size
PREPROC_DENOISE = "median"    # light denoising: "median" (3x3) | "bilateral" | "nlmeans" | "none"
PREPROC_MEDIAN_KSIZE = 3
PREPROC_CLAHE_CLIP = 2.0      # CLAHE contrast limit (0 = no CLAHE)
PREPROC_CLAHE_TILES = 8       # CLAHE grid: 8 x 8 tiles
PREPROC_JPEG_QUALITY = 95

# ================================= MODEL ====================================
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
MAX_IMAGES_PER_STUDY = 2      # frontal first, then lateral (covers ~89% of studies fully)
FUSION = "mean"               # "mean": average Q-Former outputs over a study's images
                              # "concat": keep every image's 32 query tokens (longer visual prefix)
LM_QUANTIZATION = "none"      # "none" | "8bit" | "4bit"  (4bit = QLoRA-style, needs bitsandbytes)
USE_LORA = False              # also train LoRA adapters inside the language model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_TARGET_TOKENS = 160       # 95th percentile of findings length is ~120 words

# ================================ TRAINING ==================================
EPOCHS = 4
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03
PRECISION = "auto"            # "auto" (bf16 on RTX 30xx/A100/L4 or newer, else fp16) | "fp16" | "bf16" | "fp32"
AUGMENT_TEXT_PROB = 0.0       # chance of using the paraphrased report (text_augment) as target.
                              # Off by default: the paraphrases are machine back-translations with
                              # medical-term errors ("cardiomegalia", "left lower praise collapse").
                              # Try 0.1-0.3 as an ablation only.
IMAGE_AUGMENT = True          # light rotation / crop / brightness-contrast jitter (no flips)
NUM_WORKERS = 2
GRADIENT_CHECKPOINTING = True # trades speed for memory inside the language model
MAX_VAL_STUDIES = 2000        # cap validation-loss computation per epoch (None = all)
LOG_EVERY = 50
SAVE_EVERY_STEPS = 2000       # also write last.pt every N optimizer steps (for resuming mid-epoch)
MAX_HOURS = None              # stop cleanly (saving last.pt) after this many hours of training

# ---- Kaggle overrides: T4 / P100 have 16 GB, sessions end after 12 h ----
if ON_KAGGLE:
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8          # effective batch stays 16
    NUM_WORKERS = 4
    SAVE_EVERY_STEPS = 500
    MAX_HOURS = 10.5              # leaves time for validation, samples and "Save Version"

# =============================== GENERATION =================================
MAX_NEW_TOKENS = 160
NUM_BEAMS = 4
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 0      # 0 disables
