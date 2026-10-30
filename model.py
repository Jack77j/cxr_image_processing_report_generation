"""
Multi-image BLIP-2 for study-level chest X-ray report generation.

  study images (1..N) ──► frozen ViT-g ──► Q-Former (trainable) ──► 32 query tokens / image
                                                    │
                          fusion = "mean"   : average the N sets of 32 tokens  -> 32 tokens
                          fusion = "concat" : keep all N x 32 tokens (+ learned slot embedding)
                                                    │
                          language_projection (trainable) ──► visual prefix
                                                    │
                          frozen OPT-2.7B (+ optional LoRA) ──► Findings text

Trainable by default: Q-Former, query tokens, language projection (and LoRA adapters if enabled).
Frozen: vision encoder and language model.
"""
import contextlib
import json
import os

import torch
import torch.nn as nn
import transformers
from transformers import Blip2ForConditionalGeneration, Blip2Processor

_TF_MAJOR = int(transformers.__version__.split(".")[0])
_TF_MINOR = int(transformers.__version__.split(".")[1])
_DTYPE_KW = "dtype" if (_TF_MAJOR, _TF_MINOR) >= (4, 56) else "torch_dtype"


# --------------------------------------------------------------------------- #
# precision helpers
# --------------------------------------------------------------------------- #
def resolve_precision(precision: str, device: torch.device):
    """Return (compute_dtype, use_grad_scaler)."""
    if device.type != "cuda" or precision == "fp32":
        return torch.float32, False
    if precision == "auto":
        # native bf16 needs compute capability >= 8.0 (RTX 30xx / A100 / L4 ...).
        # Turing cards (GTX 16xx, RTX 20xx, T4) only emulate it -> use fp16 there.
        major, _ = torch.cuda.get_device_capability(device)
        precision = "bf16" if major >= 8 else "fp16"
    if precision == "bf16":
        return torch.bfloat16, False
    return torch.float16, True


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class StudyBlip2(nn.Module):
    def __init__(self, blip2: Blip2ForConditionalGeneration, fusion="mean", max_images=2):
        super().__init__()
        if fusion not in ("mean", "concat"):
            raise ValueError("fusion must be 'mean' or 'concat'")
        self.blip2 = blip2
        self.fusion = fusion
        self.max_images = max_images
        self.num_query = blip2.config.num_query_tokens
        q_hidden = blip2.config.qformer_config.hidden_size
        if fusion == "concat":
            # tells the LM which image slot (1st = frontal, 2nd = lateral, ...) a token came from
            self.slot_embedding = nn.Parameter(torch.zeros(max_images, 1, q_hidden))

    # ---- helpers -----------------------------------------------------------
    @property
    def lm(self):
        return self.blip2.language_model

    def _lm_dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    def visual_prefix(self, pixel_values, image_mask):
        """
        pixel_values: (B, N, 3, H, W)   image_mask: (B, N) with 1 = real image
        returns prefix_embeds (B, P, lm_hidden), prefix_mask (B, P)
        """
        B, N = image_mask.shape
        valid = image_mask.reshape(-1).bool()
        flat = pixel_values.reshape(B * N, *pixel_values.shape[2:])[valid]      # only real images

        vis_dtype = next(self.blip2.vision_model.parameters()).dtype
        vision_trainable = any(p.requires_grad for p in self.blip2.vision_model.parameters())
        with torch.set_grad_enabled(vision_trainable and self.training):
            image_embeds = self.blip2.vision_model(pixel_values=flat.to(vis_dtype),
                                                   return_dict=True).last_hidden_state

        q_dtype = self.blip2.query_tokens.dtype
        image_embeds = image_embeds.to(q_dtype)
        enc_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.blip2.query_tokens.expand(image_embeds.shape[0], -1, -1)
        q_out = self.blip2.qformer(query_embeds=query_tokens,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=enc_mask,
                                   return_dict=True).last_hidden_state          # (n_valid, Q, H)

        Q, H = q_out.shape[1], q_out.shape[2]
        full = q_out.new_zeros(B * N, Q, H)
        full[valid] = q_out
        full = full.view(B, N, Q, H)

        if self.fusion == "mean":
            m = image_mask.view(B, N, 1, 1).to(full.dtype)
            fused = (full * m).sum(1) / m.sum(1).clamp(min=1.0)                   # (B, Q, H)
            prefix_mask = torch.ones(B, Q, dtype=torch.long, device=full.device)
        else:
            full = full + self.slot_embedding[:N].unsqueeze(0).to(full.dtype)
            fused = full.reshape(B, N * Q, H)                                     # (B, N*Q, H)
            prefix_mask = image_mask.long().repeat_interleave(Q, dim=1)

        prefix = self.blip2.language_projection(fused.to(self.blip2.language_projection.weight.dtype))
        return prefix.to(self._lm_dtype()), prefix_mask

    # ---- training ------------------------------------------------------------
    def forward(self, pixel_values, image_mask, input_ids, attention_mask, labels=None):
        prefix, prefix_mask = self.visual_prefix(pixel_values, image_mask)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, text_embeds.to(prefix.dtype)], dim=1)
        full_mask = torch.cat([prefix_mask, attention_mask.to(prefix_mask.dtype)], dim=1)
        full_labels = None
        if labels is not None:
            ignore = torch.full(prefix_mask.shape, -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([ignore, labels], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                       labels=full_labels, use_cache=False, return_dict=True)

    # ---- inference -------------------------------------------------------------
    @torch.no_grad()
    def generate(self, pixel_values, image_mask, tokenizer, max_new_tokens=160, num_beams=4,
                 repetition_penalty=1.2, no_repeat_ngram_size=0, length_penalty=1.0):
        prefix, prefix_mask = self.visual_prefix(pixel_values, image_mask)
        B = prefix.shape[0]
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=prefix.device)
        bos_embeds = self.lm.get_input_embeddings()(bos).to(prefix.dtype)
        inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)
        mask = torch.cat([prefix_mask, torch.ones_like(bos)], dim=1)
        kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
                      num_beams=num_beams, do_sample=False, repetition_penalty=repetition_penalty,
                      length_penalty=length_penalty, eos_token_id=tokenizer.eos_token_id,
                      pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id, use_cache=True)
        if no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        out = self.lm.generate(**kwargs)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in texts]


# --------------------------------------------------------------------------- #
# construction / checkpoints
# --------------------------------------------------------------------------- #
def build_model(model_name, fusion="mean", max_images=2, quantization="none", use_lora=False,
                lora_r=16, lora_alpha=32, lora_dropout=0.05, precision="auto",
                gradient_checkpointing=True, device=None):
    """Load BLIP-2, freeze/unfreeze the right parts and wrap it. Returns (model, processor, info)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype, use_scaler = resolve_precision(precision, device)
    processor = Blip2Processor.from_pretrained(model_name)

    load_kwargs = {_DTYPE_KW: compute_dtype}
    if quantization in ("8bit", "4bit"):
        if device.type != "cuda":
            raise RuntimeError("8bit/4bit quantisation needs a CUDA GPU.")
        from transformers import BitsAndBytesConfig
        skip = ["vision_model", "qformer", "language_projection", "lm_head"]
        if quantization == "4bit":
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=compute_dtype,
                                      llm_int8_skip_modules=skip)
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        load_kwargs.update(quantization_config=qcfg, device_map={"": device.index or 0})
    blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    if quantization == "none":
        blip2.to(device)

    # ---- freeze everything, then unfreeze the Q-Former path (kept in fp32) ----
    for p in blip2.parameters():
        p.requires_grad = False
    blip2.qformer.float()
    blip2.language_projection.float()
    blip2.query_tokens.data = blip2.query_tokens.data.float()
    for p in blip2.qformer.parameters():
        p.requires_grad = True
    for p in blip2.language_projection.parameters():
        p.requires_grad = True
    blip2.query_tokens.requires_grad = True

    if use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if quantization != "none":
            blip2.language_model = prepare_model_for_kbit_training(
                blip2.language_model, use_gradient_checkpointing=gradient_checkpointing)
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                          task_type="CAUSAL_LM")
        try:
            blip2.language_model = get_peft_model(blip2.language_model, lcfg)
        except ImportError as e:
            if "torchao" in str(e):
                raise ImportError(f"{e}\n-> peft is tripping over an old torchao install, which this pipeline "
                                  f"does not use. Fix: pip uninstall -y torchao") from e
            raise
        for p in blip2.language_model.parameters():     # keep adapters in fp32 (needed by GradScaler)
            if p.requires_grad:
                p.data = p.data.float()

    if gradient_checkpointing and device.type == "cuda":
        try:
            blip2.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            blip2.language_model.gradient_checkpointing_enable()

    model = StudyBlip2(blip2, fusion=fusion, max_images=max_images)
    if fusion == "concat":   # the only new parameter; the BLIP-2 weights are already placed
        model.slot_embedding.data = model.slot_embedding.data.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"device": str(device), "compute_dtype": str(compute_dtype), "use_grad_scaler": use_scaler,
            "trainable_params": trainable, "total_params": total}
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%) "
          f"| device={device} | compute dtype={compute_dtype}")
    return model, processor, info


def trainable_state_dict(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state_dict):
    own = dict(model.named_parameters())
    missing = [n for n, p in own.items() if p.requires_grad and n not in state_dict]
    unexpected = [n for n in state_dict if n not in own]
    with torch.no_grad():
        for n, t in state_dict.items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device, own[n].dtype))
    if missing or unexpected:
        print(f"[warn] checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return missing, unexpected


def load_for_inference(checkpoint_dir, weights="best.pt", device=None, precision=None):
    """Rebuild the exact model used in training from <checkpoint_dir>/run_config.json."""
    with open(os.path.join(checkpoint_dir, "run_config.json")) as f:
        run_cfg = json.load(f)
    b = run_cfg["build"]
    if precision:
        b["precision"] = precision
    model, processor, info = build_model(**{**b, "gradient_checkpointing": False, "device": device})
    ckpt = torch.load(os.path.join(checkpoint_dir, weights), map_location="cpu", weights_only=False)
    load_trainable_state(model, ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model, processor, info, run_cfg
