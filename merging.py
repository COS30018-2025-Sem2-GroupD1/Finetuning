#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merging.py  —  Offline model merging pipeline for 7B LLaMA-family models on HPC.

Use cases:
  A) Merge a LoRA adapter into the base model to produce a standalone full model.
  B) Optionally blend the result with a second full 7B model via linear interpolation (alpha merge).
  C) Optionally merge TWO LoRA adapters via task-vector arithmetic w/ trimming and DARE-style drop.

Examples
--------
# A) Merge finetuned MedAlpaca LoRA into base (pick checkpoint-2000):
python scripts/merging.py \
  --base-model-dir model/medalpaca-7b \
  --adapter-dir checkpoints/medalpaca_pubmedqa_lora \
  --adapter-step 2000 \
  --out-dir merged/medalpaca_pubmedqa_full

# B) After A, linearly blend with a second LLaMA-7B instruction model (same arch & tokenizer):
python scripts/merging.py \
  --base-model-dir model/medalpaca-7b \
  --adapter-dir checkpoints/medalpaca_pubmedqa_lora \
  --adapter-step 2000 \
  --second-model-dir /path/to/wizardlm-7b \
  --merge-mode linear --alpha 0.30 \
  --out-dir merged/medalpaca_pubmedqa_x_wizardlm_linear30

# C) Merge TWO LoRA adapters by task-vector arithmetic relative to the SAME base:
python scripts/merging.py \
  --base-model-dir model/medalpaca-7b \
  --adapter-dir checkpoints/medalpaca_pubmedqa_lora --adapter-step 2000 \
  --second-adapter-dir /path/to/generalist_lora --second-adapter-step 1000 \
  --merge-mode taskvec --alpha 0.30 \
  --trim-percentile 0.50 --dare-drop 0.50 \
  --out-dir merged/medalpaca_pubmedqa_taskvec

Notes
-----
* All merging is done on CPU by default to minimize GPU VRAM needs.
* Models MUST be architecture/tokenizer compatible for weight merges (linear/taskvec).
* For BioMistral-7B (Mistral arch) — do NOT linear/taskvec merge with LLaMA-7B. Use distillation instead.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def log(msg: str):
    print(msg, flush=True)


def device_dtype_for_hpc():
    # Do merges on CPU to save VRAM; keep fp16 weights in RAM where possible
    # but many CPUs default to float32. We'll keep tensors float32 when on CPU for safety.
    return torch.device("cpu"), torch.float32


def load_base(base_dir: Path) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    log(f"[load] base model: {base_dir}")
    tok = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.float32, device_map=None)
    # Ensure pad/eos are set
    if tok.eos_token_id is None:
        tok.add_special_tokens({'eos_token': '</s>'})
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '<|pad|>'})
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    return tok, model


def resolve_adapter_dir(adapter_root: Path, step: Optional[int]) -> Path:
    if step is None:
        # root LoRA
        return adapter_root
    sub = adapter_root / f"checkpoint-{step}"
    if (sub / "adapter_model.safetensors").exists():
        return sub
    # fallback to root if sub not found
    return adapter_root


def merge_lora_into_base(base_dir: Path, adapter_dir: Path) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok, model = load_base(base_dir)
    log(f"[merge] applying LoRA from: {adapter_dir}")
    lora = PeftModel.from_pretrained(model, str(adapter_dir))
    lora = lora.merge_and_unload()  # fold LoRA weights into base
    # ensure pad id persists
    lora.config.pad_token_id = tok.pad_token_id
    return tok, lora


def save_full_model(tok, model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[save] writing merged full model → {out_dir}")
    tok.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir), safe_serialization=True)
    log("[save] done.")


def load_full_model_for_merge(model_dir: Path) -> Tuple[AutoTokenizer, Dict[str, torch.Tensor]]:
    log(f"[load] full model for merge: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    # load to CPU, float32
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32, device_map=None)
    sd = m.state_dict()
    del m
    return tok, sd


def check_tokenizer_compat(tok_a, tok_b):
    same_vocab = (tok_a.vocab_size == tok_b.vocab_size)
    if not same_vocab:
        raise ValueError("Tokenizer vocab sizes differ; cannot merge.")
    # Optional stronger check: compare token -> id mapping of a subset
    return True


def linear_alpha_merge(sd_a: Dict[str, torch.Tensor],
                       sd_b: Dict[str, torch.Tensor],
                       alpha: float) -> Dict[str, torch.Tensor]:
    """Return (1-alpha)*A + alpha*B  (assumes identical keys & shapes)."""
    out = {}
    for k in sd_a.keys():
        if k not in sd_b:
            raise KeyError(f"Key {k} missing in second model.")
        if sd_a[k].shape != sd_b[k].shape:
            raise ValueError(f"Shape mismatch at {k}: {sd_a[k].shape} vs {sd_b[k].shape}")
        out[k] = (1.0 - alpha) * sd_a[k] + alpha * sd_b[k]
    return out


def task_vector_merge(base_sd: Dict[str, torch.Tensor],
                      delta1_sd: Dict[str, torch.Tensor],
                      delta2_sd: Dict[str, torch.Tensor],
                      alpha: float,
                      trim_percentile: float = 0.0,
                      dare_drop: float = 0.0,
                      ties_lite: bool = True) -> Dict[str, torch.Tensor]:
    """
    Merge: base + (1-alpha)*Δ1 + alpha*Δ2
    * trim_percentile: drop small deltas (by absolute magnitude) per tensor (0..1).
    * dare_drop: random-drop fraction p of remaining deltas, scale by 1/(1-p).
    * ties_lite: if signs conflict on an element, keep the larger-magnitude delta (zero the smaller).
    """
    rng = np.random.default_rng(12345)
    out = {}
    for k, base_w in base_sd.items():
        d1 = delta1_sd[k]
        d2 = delta2_sd[k]
        if d1.shape != d2.shape or d1.shape != base_w.shape:
            raise ValueError(f"Shape mismatch at {k}")

        # optional trimming per tensor
        if trim_percentile > 0.0:
            thr1 = torch.quantile(d1.abs().flatten(), torch.tensor(trim_percentile, dtype=torch.float32))
            thr2 = torch.quantile(d2.abs().flatten(), torch.tensor(trim_percentile, dtype=torch.float32))
            d1 = torch.where(d1.abs() >= thr1, d1, torch.zeros_like(d1))
            d2 = torch.where(d2.abs() >= thr2, d2, torch.zeros_like(d2))

        # optional TIES-lite sign resolution (elementwise)
        if ties_lite:
            sign_conflict = (torch.sign(d1) * torch.sign(d2)) < 0
            # where conflict: keep the one with larger |delta|
            keep_d1 = (d1.abs() >= d2.abs()) & sign_conflict
            keep_d2 = (~keep_d1) & sign_conflict
            d2 = torch.where(keep_d1, torch.zeros_like(d2), d2)
            d1 = torch.where(keep_d2, torch.zeros_like(d1), d1)
            # where no conflict -> keep both

        # weight deltas
        comb = (1.0 - alpha) * d1 + alpha * d2

        # optional DARE-style random drop
        if dare_drop > 0.0:
            mask = torch.from_numpy(rng.random(size=comb.numel()).reshape(comb.shape)).to(comb.device)
            keep = (mask >= dare_drop)
            scale = 1.0 / max(1e-6, (1.0 - dare_drop))
            comb = torch.where(keep, comb * scale, torch.zeros_like(comb))

        out[k] = base_w + comb
    return out


def state_dict_from_full_model(base_dir: Path) -> Dict[str, torch.Tensor]:
    _, m = load_base(base_dir)
    sd = m.state_dict()
    del m
    return sd


def model_from_state_dict(base_dir: Path, merged_sd: Dict[str, torch.Tensor]):
    tok, model = load_base(base_dir)
    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    if missing:
        log(f"[warn] missing keys when loading merged state: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        log(f"[warn] unexpected keys when loading merged state: {len(unexpected)} (first 5): {unexpected[:5]}")
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, help="Path to base LLaMA-7B (HF format)")
    ap.add_argument("--adapter-dir", required=True, help="Path to first LoRA adapter root")
    ap.add_argument("--adapter-step", type=int, default=None, help="Pick checkpoint-<step> inside adapter-dir (optional)")

    ap.add_argument("--merge-mode", choices=["lora_only","linear","taskvec"], default="lora_only",
                    help="lora_only: just fold LoRA into base; "
                         "linear: blend full merged model with second full model; "
                         "taskvec: merge TWO LoRA adapters relative to base")

    # For linear model-model merge
    ap.add_argument("--second-model-dir", default=None, help="Path to second FULL model (same arch/tokenizer)")
    ap.add_argument("--alpha", type=float, default=0.30, help="Blend weight for second model/adapter (0..1)")

    # For task-vector merge of TWO adapters
    ap.add_argument("--second-adapter-dir", default=None, help="Path to second LoRA adapter root")
    ap.add_argument("--second-adapter-step", type=int, default=None, help="checkpoint-<step> inside second adapter")
    ap.add_argument("--trim-percentile", type=float, default=0.0, help="Per-tensor abs-delta trim percentile (0..1)")
    ap.add_argument("--dare-drop", type=float, default=0.0, help="Random drop rate (0..1) with 1/(1-p) rescale")
    ap.add_argument("--no-ties-lite", action="store_true", help="Disable TIES-lite sign conflict handling")

    ap.add_argument("--out-dir", required=True, help="Where to write merged full model")
    args = ap.parse_args()

    base_dir = Path(args.base_model_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Always produce FULL model from first adapter
    adapter_dir = resolve_adapter_dir(Path(args.adapter_dir), args.adapter_step)
    tok_full, model_full = merge_lora_into_base(base_dir, adapter_dir)
    tmp_full_dir = out_dir / "_tmp_full_from_adapter"
    save_full_model(tok_full, model_full, tmp_full_dir)

    if args.merge_mode == "lora_only":
        # finalize: move tmp as final (or re-save to clean dir)
        log("[done] LoRA merged into base. No further merging requested.")
        return

    # Step 2: Additional merging
    if args.merge_mode == "linear":
        if not args.second_model_dir:
            raise ValueError("--second-model-dir is required for linear merge")
        # load both full models’ states
        tok_a, sd_a = load_full_model_for_merge(tmp_full_dir)
        tok_b, sd_b = load_full_model_for_merge(Path(args.second_model_dir))
        check_tokenizer_compat(tok_a, tok_b)
        log(f"[linear] alpha={args.alpha:.2f} → merged = (1-α)*A + α*B")
        merged_sd = linear_alpha_merge(sd_a, sd_b, args.alpha)
        tok_out, model_out = model_from_state_dict(base_dir, merged_sd)
        save_full_model(tok_out, model_out, out_dir)
        # clean temp
        return

    if args.merge_mode == "taskvec":
        if not args.second_adapter_dir:
            raise ValueError("--second-adapter-dir is required for taskvec merge")
        # Build two full models (fold each LoRA into base) and compute deltas to base
        # (We already have full A at tmp_full_dir)
        tok_a, sd_a = load_full_model_for_merge(tmp_full_dir)

        second_adapter_dir = resolve_adapter_dir(Path(args.second_adapter_dir), args.second_adapter_step)
        # produce full model for second adapter into another tmp dir
        tmp_full_dir_b = out_dir / "_tmp_full_from_adapter_b"
        _, model_full_b = merge_lora_into_base(base_dir, second_adapter_dir)
        save_full_model(tok_a, model_full_b, tmp_full_dir_b)
        tok_b, sd_b = load_full_model_for_merge(tmp_full_dir_b)

        check_tokenizer_compat(tok_a, tok_b)

        base_sd = state_dict_from_full_model(base_dir)

        # deltas
        delta1 = {k: sd_a[k] - base_sd[k] for k in base_sd.keys()}
        delta2 = {k: sd_b[k] - base_sd[k] for k in base_sd.keys()}

        log(f"[taskvec] alpha={args.alpha:.2f}, trim={args.trim_percentile:.2f}, dare_drop={args.dare_drop:.2f}, ties_lite={not args.no_ties_lite}")
        merged_sd = task_vector_merge(
            base_sd, delta1, delta2, alpha=args.alpha,
            trim_percentile=args.trim_percentile,
            dare_drop=args.dare_drop,
            ties_lite=(not args.no_ties_lite)
        )
        tok_out, model_out = model_from_state_dict(base_dir, merged_sd)
        save_full_model(tok_out, model_out, out_dir)
        return


if __name__ == "__main__":
    main()
