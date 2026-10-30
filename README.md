# Study-level CXR Findings Generation (BLIP-2, multi-image) — image pre-processing variant

> **This folder is a copy of `StudyLevel_Pipeline` with one added step: image pre-processing
> (light denoising + CLAHE) before training.** Everything else (data preparation, 25k subset, model,
> training settings, generation, metrics) is unchanged, so results compare directly with the
> original run `subset25k_v1`. See "Image pre-processing" below.

Generates the **Findings** section for one chest X-ray **study**: all X-rays from a single visit
(usually a frontal view, sometimes with a lateral). At use time, running it on a patient's
current study gives **one report per patient that describes their current state**.

Why study level: the CSVs don't record when studies happened (study IDs are random), so you
can't tell which study is a patient's latest or work out change over time. Within one study,
though, images and report are paired exactly.

## Methodology

```
patient-level CSV ─► prepare_data.py ─► one row per STUDY
                       • images grouped by study id, paired with that study's report
                       • image order: PA, AP, LATERAL, OTHER  (frontal first)
                       • target = Findings (fallback: Impression)
                       • drop sentences about earlier scans / phoning clinicians
                       • split train/val by subject_id; official validate CSV = test

study images (≤2) ─► frozen ViT-g ─► Q-Former (trained) ─► 32 query tokens per image
                                   ─► fusion: mean (default) or concat
                                   ─► language projection (trained)
                                   ─► frozen OPT-2.7B (+ optional LoRA) ─► Findings text
```

Trainable parts: the Q-Former, the query tokens and the language projection (the exact
parameter count is printed when training starts). LoRA on the language model is optional (`--use_lora`). For a QLoRA-style setup
that fits smaller GPUs, add `--use_lora --quantization 4bit`.

## Image pre-processing (step 1c, `preprocess_images.py`)

Run once after `select_subset.py`, before training:

```
python preprocess_images.py --csv outputs/processed/train_subset_25k.csv outputs/processed/val.csv outputs/processed/test.csv --max_images 1
```

Per image (grayscale):

1. downscale so the longest side is at most 512 px (aspect ratio kept; the model input is 224 × 224 anyway)
2. **light denoising**: 3 × 3 median filter (`PREPROC_DENOISE`; also `bilateral`, `nlmeans`, `none`)
3. **CLAHE** contrast enhancement: clip limit 2.0, 8 × 8 tiles (`PREPROC_CLAHE_CLIP`, `PREPROC_CLAHE_TILES`).
   Denoising comes first because CLAHE would otherwise amplify the noise.
4. saved as JPEG (quality 95) under `outputs/images_proc/` with the same relative path as the original

* The **training subset, validation and test images all get the same processing**, so the model sees the
  same kind of image at test time as in training. The original dataset images are only read.
* Only the images that are used are processed (the first frontal image of each study with `--max_images 1`):
  about 25,000 + 7,101 + 1,172 images.
* `train.py` and `generate.py` read from `config.PROC_IMAGE_ROOT` by default (`--image_root` overrides).
  `generate.py --images ...` applies the same processing to raw image files on the fly.
* The augmentation during training (small rotation, crop, brightness/contrast) and the BLIP-2 image
  processor (resize to 224 × 224, normalise) are unchanged and run on top of the processed images.
* Already processed files are skipped (the step can be resumed). Changing the settings needs `--overwrite`.
* Outputs: `outputs/images_proc/preprocessing_summary.json` and `outputs/figures/preprocessing_examples.png`
  (original | denoised | denoised + CLAHE).

## Files

| Step | File | What it does |
|---|---|---|
| – | `config.py` | **Edit the paths here first** (on Kaggle they are found automatically). All defaults live here. |
| 1 | `prepare_data.py` | Patient CSVs → `outputs/processed/{train,val,test}.csv` + `data_summary.json` |
| – | `text_cleaning.py` | Section splitting, comparison and communication sentence removal |
| 1b | `select_subset.py` | Optional: balanced, quality-filtered training subset with a fixed image budget (e.g. 25,000) |
| 1c | `preprocess_images.py` | **New:** light denoising + CLAHE for the subset, val and test images → `outputs/images_proc/` |
| 2 | `eda.py` | Figures for the report: images per study, views, target length, finding prevalence |
| – | `dataset.py` | `StudyCXRDataset`: padded, masked image stack + tokenised target (with EOS) |
| – | `model.py` | `StudyBlip2` wrapper, freezing, LoRA/quantisation, checkpoint helpers |
| 3 | `train.py` | Training with AMP, grad accumulation, best/last checkpoints, `--resume` |
| 4 | `generate.py` | Predictions for a whole split, or a report for one study folder / image set |
| 5 | `evaluate.py` | BLEU-1..4, METEOR, ROUGE-L, CIDEr-D and clinical P/R/F1 (14 + 6 key findings) |
| – | `clinical_labeler.py` | Rule-based CheXpert-style labeler used for the clinical metrics |
| – | `run_all.bat` | Runs steps 1–5 in order on Windows |
| – | `kaggle_notebook.ipynb` | Runs steps 1–5 on a Kaggle GPU, continuing across 12-hour sessions |

## Balanced training subset (e.g. 25,000 images)

`select_subset.py` builds a smaller training set from `processed/train.csv`. The val and test sets are **not** changed, so results stay comparable. It picks studies as follows:

1. **Quality filters.** A study must contain a frontal (PA/AP) image, and its cleaned report must be 8–120 words long. This removes one-liners such as "No acute process." and runaway reports.
2. **Patient diversity.** At most 2 studies per patient.
3. **Finding balance.** Each report is labelled with the 14 CheXpert findings. Studies are sampled with weight `(1 / frequency of their rarest finding) ** 0.5`, so rare findings are picked more often. Normal studies are capped at 30% (46% in the full set). Reports whose target fell back to the Impression section are picked half as often, because Findings text is more descriptive.
4. **Exact image budget.** The default is one frontal image per study, which gives 25,000 distinct reports for 25,000 images. `--images_per_study 2` gives frontal + lateral pairs instead (about 16,000 studies).

What it produces on the Kaggle data (25,000 images):

| | Full train (after filters) | 25k subset |
|---|---|---|
| Studies / patients | 123,743 / 40,464 | 25,000 / 18,435 |
| Normal ("No Finding") | 46.0% | 30.0% |
| Cardiomegaly / Edema / Effusion | 12.3 / 11.2 / 18.9% | 16.0 / 13.9 / 21.5% |
| Consolidation / Pneumothorax / Lesion | 3.5 / 2.2 / 2.8% | 6.2 / 3.9 / 6.2% |
| Target from Findings section | 71% | 84% |

```bat
python select_subset.py --images 25000
python train.py --train_csv outputs\processed\train_subset_25k.csv --max_images 1 --use_lora --run_name subset25k_v1
```

It also writes `train_subset_25k_summary.json` and a before/after prevalence chart in `outputs/figures/`. The trade-off is that the subset has more abnormal cases than real life, so the model mentions findings more readily. That usually raises clinical recall/F1 but can add some false positives.

## Running on Kaggle (recommended: free 16 GB GPU)

One-time setup:

1. **Upload the code:** *Datasets → New Dataset*, drag in this `StudyLevel_Pipeline` folder, and keep it private. Re-upload it (*New Version*) whenever the code changes.
2. **Create the notebook:** *Code → New Notebook → File → Import Notebook*, and choose `kaggle_notebook.ipynb`.
3. **Notebook settings:** Accelerator **GPU T4 x2**, and **Internet On**. Internet access needs a phone-verified Kaggle account, and is required to download BLIP-2.
4. **Add Input:** the MIMIC-CXR dataset with `mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv` and `official_data_iccv_final/`, plus your code dataset from step 1.

Each session:

- Set `RUN_NAME` and the other options in the first code cell, then run **Save Version → Save & Run All (Commit)**. This runs in the background for up to 12 hours.
- `config.py` finds the data under `/kaggle/input` by itself and writes everything to `/kaggle/working/outputs`.
- On Kaggle it uses batch 2 × accumulation 8 (effective batch 16) and 4 data workers, and stops training cleanly after **10.5 hours** (`MAX_HOURS`). Before stopping it runs validation, updates `best.pt` if improved, saves `last.pt`, and writes `STATUS.txt = incomplete`.
- **To continue in the next session:** *Add Input → Notebook Output →* this notebook's latest version, then *Save & Run All* again. The notebook reuses the processed CSVs, resumes from `last.pt` at the exact batch where it stopped, and runs test-set generation and evaluation once `STATUS.txt` says `complete`.
- Results to download: `outputs/results/<run>/metrics.json`, `clinical_per_class.csv`, `predictions_test.csv`, and `outputs/figures/`.
- **Check an unfinished run:** set `EVAL_ONLY = True` (and optionally `MAX_TEST_STUDIES = 300`), keep the previous output as an input, and run. Training is skipped. The current `best.pt` (or `EVAL_WEIGHTS = "last.pt"`) is scored, and the results go to `outputs/results/<run>/eval_best/`. The run folder is copied into this version's output, so you can resume from that output next time with `EVAL_ONLY = False`.
- `evaluate.py` also prints a **Diversity** block: the number of unique reports, and the share of studies where a key finding is mentioned (generated vs. reference). If that share stays near 0%, the model is writing the same "normal" report for everyone.

Command-line equivalent for a new session: `python train.py --resume_from /kaggle/input/<previous-output>/outputs/checkpoints/<run>`.

## Setup

```bat
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   :: pick your CUDA version
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

## Run

```bat
python prepare_data.py                         :: ~minutes; checks every image exists on disk
python eda.py --split train
python train.py --max_train_studies 20000 --epochs 2 --run_name pilot     :: quick first run
python train.py --run_name mean_v1             :: full run
python generate.py --run_dir outputs\checkpoints\mean_v1
python evaluate.py --predictions outputs\results\mean_v1\predictions_test.csv
```

Other runs:

```bat
:: one study (put the frontal image first)
python generate.py --run_dir outputs\checkpoints\mean_v1 --study_dir "D:\BioMed\archive (5)\official_data_iccv_final\files\p10\p10003502\s51180958"

:: resume an interrupted run (continues mid-epoch from the last save)
python train.py --resume --run_name mean_v1

:: ablations for the results table
python train.py --fusion concat --run_name concat_v1
python train.py --use_lora --run_name mean_lora_v1
python train.py --use_lora --quantization 4bit --batch_size 2 --grad_accum_steps 8 --run_name qlora_v1
python prepare_data.py --comparison_mode keep --out_dir outputs\processed_keep   :: effect of cleaning
```

## GPU memory (rough guide)

- The OPT-2.7B BLIP-2 weights alone take about 8 GB in fp16.
- With the defaults (batch 4, 2 images, gradient checkpointing), plan for about 16 GB+.
- On 12 GB, use `--batch_size 2 --grad_accum_steps 8`, and add `--quantization 8bit` or `4bit` if it still runs out of memory.
- Each epoch over the full training set (roughly 200k studies) takes a long time on one GPU. Start with `--max_train_studies`, and check `samples_epoch*.txt` after each epoch.
- Precision is chosen automatically. bf16 is used on RTX 30xx, A100, L4 and newer. fp16 is used on older cards (T4, P100, GTX 16xx, RTX 20xx).
- If a run stops before its first full epoch ends, `generate.py` falls back to `last.pt` when there is no `best.pt` yet.

## Notes and limitations

- **Comparison sentences:** In `drop` mode, sentences like "unchanged pleural effusion" are removed entirely, so some real findings go with them. The trade-off is that the model doesn't learn to invent references to earlier scans. Compare against `--comparison_mode keep` in your results.
- **`text_augment`:** This column is a machine back-translation and contains medical-term errors ("cardiomegalia", "left lower praise collapse"). It is off by default (`AUGMENT_TEXT_PROB = 0.0`).
- **Clinical metrics:** The built-in labeler is a keyword and negation approximation of CheXpert. Use it to compare runs against each other. For publication numbers, add `--chexbert` / `--radgraph` if those packages install.
- This is a research and educational pipeline, not a diagnostic tool.
