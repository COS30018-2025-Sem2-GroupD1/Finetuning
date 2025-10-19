#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merging.py — Offline FULL-model merging for LLaMA-7B–family checkpoints on HPC.

Scenarios (no adapters assumed):
A) Linear blend two FULL models (same arch/tokenizer) with alpha.
B) Task-vector merge TWO FULL models relative to an ANCHOR base:
     θ_out = θ_anchor + (1-α)·(θ_A - θ_anchor) + α·(θ_B - θ_anchor)
   with optional per-tensor |Δ|-trim, TIES-lite sign gating, and DARE drop.

NOTES
-----
* MedGemma-27B-IT (Gemma arch, 27B, different tokenizer) is NOT compatible for raw weight merging.
  Use knowledge distillation, RAG, or training instead.
* For LLaMA-7B family donors (same tokenizer), you may include:
  - MedAlpaca-7B derivatives (your 'model/medalpaca-full')
  - Meditron-7B, PMC-LLaMA-7B, ClinicalCamel-7B (verify tokenizer/arch match!)
* If tokenizers differ but architectures match, you can try --exclude-embeddings to keep
  the primary model's embeddings & lm_head intact (safer, but still risky).

EXAMPLES
--------
# 1) Linear blend your merged MedAlpaca with another LLaMA-7B model:
python scripts/merging.py \
  --mode linear \
  --primary-model-dir model/medalpaca-full \
  --second-model-dir /path/to/other-llama7b \
  --alpha 0.30 \
  --out-dir merged/medalpaca_full_x_other_linear30

# 2) Task-vector merge (anchor = original MedAlpaca-7B base), combining TWO full models:
python scripts/merging.py \
  --mode taskvec \
  --anchor-model-dir model/medalpaca-7b \
  --primary-model-dir model/medalpaca-full \
  --second-model-dir /path/to/pmc-llama-7b \
  --alpha 0.35 \
  --trim-percentile 0.50 --dare-drop 0.50 \
  --out-dir merged/medalpaca_full_taskvec_pmc_trim50_dare50

# 3) Same as #2 but skip embedding & head keys if tokenizers differ:
python scripts/merging.py \
  --mode taskvec \
  --anchor-model-dir model/medalpaca-7b \
  --primary-model-dir model/medalpaca-full \
  --second-model-dir /path/to/clinicalcamel-7b \
  --alpha 0.20 \
  --exclude-embeddings \
  --out-dir merged/medalpaca_full_taskvec_clinical_excl_emb

Best practices
--------------
* Do merges on CPU in float32 (default). Cast to bf16 for serving right before saving.
* Lock decoding/eval settings; measure per-subtask to catch regressions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig


# ------------------------------- logging ------------------------------------ #

def log(msg: str):
    print(msg, flush=True)


# ------------------------------- I/O helpers -------------------------------- #

def load_tokenizer(path: Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(str(path), use_fast=True)


def load_config(path: Path) -> PretrainedConfig:
    return PretrainedConfig.from_pretrained(str(path))


def load_full_model_state(path: Path, dtype=torch.float32) -> Tuple[AutoTokenizer, Dict[str, torch.Tensor], PretrainedConfig]:
    """Instantiate the HF model on CPU to obtain a canonical state_dict."""
    log(f"[load] model: {path}")
    tok = load_tokenizer(path)
    cfg = load_config(path)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    return tok, sd, cfg


def instantiate_like(reference_dir: Path, dtype=torch.float32):
    """Create a fresh model instance (matching architecture of reference_dir) for loading a merged state."""
    tok = load_tokenizer(reference_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(reference_dir),
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # ensure pad/eos presence
    if tok.eos_token_id is None:
        tok.add_special_tokens({'eos_token': '</s>'})
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '<|pad|>'})
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    if hasattr(model, "config"):
        model.config.pad_token_id = tok.pad_token_id
        model.generation_config.pad_token_id = tok.pad_token_id if hasattr(model, "generation_config") else tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id if hasattr(model, "generation_config") else tok.eos_token_id
    return tok, model


def save_full_model(tok: AutoTokenizer, model: AutoModelForCausalLM, out_dir: Path, save_dtype: str = "bf16"):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(save_dtype, torch.bfloat16)
    log(f"[save] casting to {save_dtype} and writing → {out_dir}")
    model = model.to(to_dtype)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    with open(out_dir / "MERGE_INFO.txt", "w") as f:
        f.write(json.dumps({"save_dtype": save_dtype}, indent=2))
    log("[save] done.")


# -------------------------- compatibility checks --------------------------- #

def same_arch(cfg_a: PretrainedConfig, cfg_b: PretrainedConfig) -> bool:
    keys = [
        "model_type", "hidden_size", "num_attention_heads",
        "num_hidden_layers", "intermediate_size"
    ]
    return all(getattr(cfg_a, k, None) == getattr(cfg_b, k, None) for k in keys)


def tokenizers_compatible(tok_a: AutoTokenizer, tok_b: AutoTokenizer) -> bool:
    # strict check: same vocab size and same special token ids
    same_vocab = tok_a.vocab_size == tok_b.vocab_size
    same_eos = tok_a.eos_token_id == tok_b.eos_token_id
    same_pad = tok_a.pad_token_id == tok_b.pad_token_id
    return bool(same_vocab and same_eos and same_pad)


# --------------------------- key filtering utils --------------------------- #

EMBED_KEYS = ("model.embed_tokens.weight",)
HEAD_KEYS  = ("lm_head.weight",)

def iter_layer_keys(sd: Dict[str, torch.Tensor], layer_range: Optional[str]) -> Iterable[str]:
    """Yield keys restricted to an optional LLaMA layer range like '0:32' (start:end)."""
    if not layer_range:
        for k in sd.keys():
            yield k
        return
    start, end = layer_range.split(":")
    start_i = int(start) if start != "" else 0
    end_i   = int(end)   if end   != "" else 10**9
    # Include non-layer weights (embeddings/head/norms) only if they don't encode a layer index
    for k in sd.keys():
        if ".layers." in k:
            # extract ...layers.<idx>...
            try:
                seg = k.split(".layers.")[1]
                idx = int(seg.split(".")[0])
                if start_i <= idx < end_i:
                    yield k
            except Exception:
                # if parsing fails, include conservatively
                yield k
        else:
            # always yield non-layer keys; they can be filtered by exclude flags below
            yield k


def filter_merge_keys(keys: Iterable[str], exclude_embeddings: bool, exclude_lm_head: bool) -> Iterable[str]:
    for k in keys:
        if exclude_embeddings and any(k == ek or k.startswith("model.embed_tokens") for ek in EMBED_KEYS):
            continue
        if exclude_lm_head and any(k == hk or k.startswith("lm_head") for hk in HEAD_KEYS):
            continue
        yield k


# ----------------------------- merge kernels ------------------------------ #

def linear_alpha_merge(sd_a: Dict[str, torch.Tensor],
                       sd_b: Dict[str, torch.Tensor],
                       keys: Iterable[str],
                       alpha: float) -> Dict[str, torch.Tensor]:
    out = {}
    for k in keys:
        if k not in sd_a or k not in sd_b:
            continue
        if sd_a[k].shape != sd_b[k].shape:
            raise ValueError(f"[linear] shape mismatch at {k}: {sd_a[k].shape} vs {sd_b[k].shape}")
        out[k] = (1.0 - alpha) * sd_a[k] + alpha * sd_b[k]
    return out


def compute_delta(anchor: Dict[str, torch.Tensor],
                  target: Dict[str, torch.Tensor],
                  keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k in keys:
        if k in anchor and k in target and anchor[k].shape == target[k].shape:
            delta[k] = target[k] - anchor[k]
    return delta


def apply_delta(anchor: Dict[str, torch.Tensor],
                delta: Dict[str, torch.Tensor],
                keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in keys:
        if k in anchor:
            if k in delta:
                out[k] = anchor[k] + delta[k]
            else:
                out[k] = anchor[k]
    return out


def trim_by_percentile(delta: Dict[str, torch.Tensor], percentile: float) -> Dict[str, torch.Tensor]:
    if percentile <= 0.0:
        return delta
    trimmed = {}
    q = float(percentile)
    for k, v in delta.items():
        thr = torch.quantile(v.abs().flatten(), torch.tensor(q, dtype=torch.float32))
        trimmed[k] = torch.where(v.abs() >= thr, v, torch.zeros_like(v))
    return trimmed


def ties_lite_merge(d1: Dict[str, torch.Tensor],
                    d2: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Elementwise sign conflict resolution: keep larger |Δ| where signs disagree."""
    out1, out2 = {}, {}
    for k in d1.keys():
        a = d1[k]; b = d2[k]
        if a.shape != b.shape:
            raise ValueError(f"[ties] shape mismatch at {k}")
        conflict = (torch.sign(a) * torch.sign(b)) < 0
        keep_a = (a.abs() >= b.abs()) & conflict
        keep_b = (~keep_a) & conflict
        b2 = torch.where(keep_a, torch.zeros_like(b), b)
        a2 = torch.where(keep_b, torch.zeros_like(a), a)
        out1[k] = a2
        out2[k] = b2
    return out1, out2


def dare_drop(delta: Dict[str, torch.Tensor], drop: float, seed: int = 12345) -> Dict[str, torch.Tensor]:
    if drop <= 0.0:
        return delta
    rng = np.random.default_rng(seed)
    out = {}
    keep_scale = 1.0 / max(1e-6, (1.0 - drop))
    for k, v in delta.items():
        mask = torch.from_numpy(rng.random(size=v.numel()).reshape(v.shape)).to(v.device)
        keep = (mask >= drop)
        out[k] = torch.where(keep, v * keep_scale, torch.zeros_like(v))
    return out


# ------------------------------- main logic -------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear", "taskvec"], required=True)

    # paths
    ap.add_argument("--primary-model-dir", required=True, help="FULL model A (e.g., model/medalpaca-full)")
    ap.add_argument("--second-model-dir", required=True, help="FULL model B (same arch; tokenizer ideally same)")
    ap.add_argument("--anchor-model-dir", default=None, help="[taskvec] Anchor/base model for deltas (e.g., model/medalpaca-7b)")

    # knobs
    ap.add_argument("--alpha", type=float, default=0.30, help="Blend weight for second model (0..1)")
    ap.add_argument("--layer-range", default=None, help="Optional LLaMA layer range 'start:end' (e.g., '16:32')")
    ap.add_argument("--exclude-embeddings", action="store_true", help="Skip model.embed_tokens.* from merge")
    ap.add_argument("--exclude-lm-head", action="store_true", help="Skip lm_head.* from merge")
    ap.add_argument("--trim-percentile", type=float, default=0.0, help="[taskvec] Per-tensor |Δ| trim (0..1)")
    ap.add_argument("--dare-drop", type=float, default=0.0, help="[taskvec] DARE random drop (0..1)")
    ap.add_argument("--no-ties-lite", action="store_true", help="[taskvec] Disable TIES-lite sign gating")

    # save
    ap.add_argument("--save-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--out-dir", required=True, help="Output dir for merged model")
    args = ap.parse_args()

    primary_dir = Path(args.primary-model-dir)
    second_dir  = Path(args.second-model-dir)
    out_dir     = Path(args.out-dir)

    # load
    tok_a, sd_a, cfg_a = load_full_model_state(primary_dir)
    tok_b, sd_b, cfg_b = load_full_model_state(second_dir)

    # arch check
    if not same_arch(cfg_a, cfg_b):
        raise SystemExit("[ERR] Models differ in architecture; abort.")

    # tokenizer check (strict only if not excluding embeddings & lm_head)
    strict_tok = not (args.exclude-embeddings or args.exclude-lm_head)
    if strict_tok and not tokenizers_compatible(tok_a, tok_b):
        raise SystemExit("[ERR] Tokenizers differ; use --exclude-embeddings/--exclude-lm-head or choose compatible donors.")
    elif not strict_tok and not tokenizers_compatible(tok_a, tok_b):
        log("[warn] Tokenizers differ; embeddings / lm_head will be preserved from primary (excluded from merge).")

    # select keys
    keys_iter = iter_layer_keys(sd_a, args.layer-range)
    keys_iter = filter_merge_keys(keys_iter, args.exclude-embeddings, args.exclude-lm_head)
    keys = list(keys_iter)

    if args.mode == "linear":
        log(f"[linear] α={args.alpha:.2f} on {len(keys)} params (excl embeddings={args.exclude-embeddings}, lm_head={args.exclude-lm_head})")
        merged_sd = dict(sd_a)  # start from A
        blended = linear_alpha_merge(sd_a, sd_b, keys, args.alpha)
        merged_sd.update(blended)

        # instantiate and save
        tok_out, model_out = instantiate_like(primary_dir, dtype=torch.float32)
        missing, unexpected = model_out.load_state_dict(merged_sd, strict=False)
        if missing:   log(f"[warn] missing keys: {len(missing)} (first 5): {missing[:5]}")
        if unexpected:log(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
        save_full_model(tok_out, model_out, out_dir, save_dtype=args.save_dtype)
        return

    # taskvec
    if args.mode == "taskvec":
        if not args.anchor_model-dir:
            raise SystemExit("[ERR] --anchor-model-dir is required for taskvec mode.")
        anchor_dir = Path(args.anchor-model-dir)
        _, sd_anchor, cfg_anchor = load_full_model_state(anchor_dir)
        if not same_arch(cfg_a, cfg_anchor) or not same_arch(cfg_b, cfg_anchor):
            raise SystemExit("[ERR] Anchor architecture must match both donors.")

        log(f"[taskvec] building deltas vs anchor on {len(keys)} keys…")
        delta_a = compute_delta(sd_anchor, sd_a, keys)
        delta_b = compute_delta(sd_anchor, sd_b, keys)

        if not args.no_ties_lite:
            delta_a, delta_b = ties_lite_merge(delta_a, delta_b)

        if args.trim_percentile > 0.0:
            delta_a = trim_by_percentile(delta_a, args.trim_percentile)
            delta_b = trim_by_percentile(delta_b, args.trim_percentile)

        # combine
        comb = {}
        for k in keys:
            da = delta_a.get(k, None)
            db = delta_b.get(k, None)
            if da is None and db is None:
                continue
            if da is None:
                comb[k] = args.alpha * db
            elif db is None:
                comb[k] = (1.0 - args.alpha) * da
            else:
                comb[k] = (1.0 - args.alpha) * da + args.alpha * db

        if args.dare_drop > 0.0:
            comb = dare_drop(comb, args.dare_drop)

        merged_sd = dict(sd_anchor)
        for k, v in comb.items():
            merged_sd[k] = sd_anchor[k] + v

        tok_out, model_out = instantiate_like(anchor_dir, dtype=torch.float32)
        missing, unexpected = model_out.load_state_dict(merged_sd, strict=False)
        if missing:   log(f"[warn] missing keys: {len(missing)} (first 5): {missing[:5]}")
        if unexpected:log(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
        save_full_model(tok_out, model_out, out_dir, save_dtype=args.save_dtype)
        return


if __name__ == "__main__":
    main()
