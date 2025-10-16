#!/usr/bin/env python3

'''
SFT_CKPT=checkpoint/medalpaca_pubmedqa_lora/checkpoint-2000
KD_CKPT=checkpoint/medalpaca_pubmedqa_lora_v2/checkpoint-9974
python scripts/merge_export_lora.py \
  --base model/medalpaca-7b \
  --sft_adapter "$SFT_CKPT" \
  --kd_adapter  "$KD_CKPT" \
  --model_dir model \
  --name_full medalpaca-full \
  --name_sft  medalpaca-sft \
  --name_kd   medalpaca-kd \
  --prefer_bf16

'''

import os, sys, argparse, shutil, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

torch.set_float32_matmul_precision("high")

def ensure_tok(tok):
    # Make sure pad/eos exist and are consistent
    added = False
    if tok.eos_token_id is None and tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"}); added = True
    if tok.pad_token_id is None and tok.pad_token is None:
        # prefer eos as pad for causal LM
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        added = True
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return added

def load_base(base_dir, prefer_bf16=True):
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True, trust_remote_code=True)
    ensure_tok(tok)
    # Load base in a real dtype (not 4/8-bit) so we can merge LoRA into full weights
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Ensure generation config has pad/eos
    if hasattr(model, "generation_config") and model.generation_config:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
    return tok, model

def merge_adapter_into(model, adapter_dir):
    # Load PEFT adapter and merge into the base weights
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    peft_model = peft_model.merge_and_unload()   # <- merges LoRA and returns a plain HF model
    return peft_model

def save_model_and_tokenizer(model, tokenizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # If tokenizer grew special tokens, resize embeddings
    if getattr(model, "get_input_embeddings", None) and tokenizer.vocab_size != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # Save
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # Touch a README hint
    with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as f:
        f.write("Merged LoRA adapters into base (PEFT merge_and_unload). Saved with safetensors.\n")

def make_symlink_or_copy(src_dir, dst_dir, copy=False):
    dst = pathlib.Path(dst_dir)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_dir():
            if dst.is_symlink() or len(list(dst.glob("*"))) == 0:
                dst.unlink(missing_ok=True)
            else:
                raise SystemExit(f"[ERR] Destination exists and is not empty: {dst}")
    if copy:
        shutil.copytree(src_dir, dst_dir)
    else:
        # create a relative symlink
        rel = os.path.relpath(src_dir, os.path.dirname(dst_dir))
        os.symlink(rel, dst_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base model (e.g., model/medalpaca-7b)")
    ap.add_argument("--sft_adapter", required=True, help="Path to SFT LoRA checkpoint dir (e.g., checkpoint/.../checkpoint-2000)")
    ap.add_argument("--kd_adapter",  required=True, help="Path to KD  LoRA checkpoint dir (e.g., checkpoint/.../checkpoint-9974)")
    ap.add_argument("--model_dir",   default="model", help="Output parent dir (default: model)")
    ap.add_argument("--name_full",   default="medalpaca-full")
    ap.add_argument("--name_sft",    default="medalpaca-sft")
    ap.add_argument("--name_kd",     default="medalpaca-kd")
    ap.add_argument("--copy_full",   action="store_true", help="Copy base to medalpaca-full instead of symlink")
    ap.add_argument("--prefer_bf16", action="store_true", help="Prefer bfloat16 if available")
    args = ap.parse_args()

    base_dir = args.base
    out_full = os.path.join(args.model_dir, args.name_full)
    out_sft  = os.path.join(args.model_dir, args.name_sft)
    out_kd   = os.path.join(args.model_dir, args.name_kd)

    print(f"[INFO] Linking/copying base -> {out_full}")
    make_symlink_or_copy(base_dir, out_full, copy=args.copy_full)

    print(f"[INFO] Loading base: {base_dir}")
    tok, base_model = load_base(base_dir, prefer_bf16=args.prefer_bf16)

    # --- SFT merge ---
    print(f"[INFO] Merging SFT adapter: {args.sft_adapter}")
    sft_merged = merge_adapter_into(base_model, args.sft_adapter)
    print(f"[INFO] Saving merged SFT -> {out_sft}")
    save_model_and_tokenizer(sft_merged, tok, out_sft)
    # free memory
    del sft_merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reload clean base for the next merge
    print(f"[INFO] Reloading base for KD merge...")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _, base_model = load_base(base_dir, prefer_bf16=args.prefer_bf16)

    # --- KD merge ---
    print(f"[INFO] Merging KD adapter: {args.kd_adapter}")
    kd_merged = merge_adapter_into(base_model, args.kd_adapter)
    print(f"[INFO] Saving merged KD -> {out_kd}")
    save_model_and_tokenizer(kd_merged, tok, out_kd)

    print("\n[OK] Done.")
    print(f" - Base (full): {out_full} (symlink{' copy' if args.copy_full else ''})")
    print(f" - SFT merged : {out_sft}")
    print(f" - KD  merged : {out_kd}")

if __name__ == "__main__":
    main()
