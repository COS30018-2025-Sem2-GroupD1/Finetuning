#!/usr/bin/env python
import os, sys, argparse, json
from pathlib import Path

'''
python scripts/merge_reranker_lora.py \
  --base model/bge-reranker-v2-gemma \
  --lora outputs/reranker-medical-gemma/checkpoint-22000 \
  --out  outputs/reranker-medical-gemma-merged \
  --dtype float16
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model dir (e.g., 
model/bge-reranker-v2-gemma)")
    ap.add_argument("--lora", required=True, help="LoRA checkpoint dir 
(e.g., outputs/.../checkpoint-22000)")
    ap.add_argument("--out",  required=True, help="Output merged dir 
(e.g., outputs/reranker-medical-gemma-merged)")
    ap.add_argument("--dtype", default="float16", 
choices=["float16","bfloat16","float32"])
    ap.add_argument("--device", default="auto", 
choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    base_dir = Path(args.base)
    lora_dir = Path(args.lora)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dtype
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, 
"float32": torch.float32}[args.dtype]

    print(f"[merge] Loading base: {base_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=dtype, low_cpu_mem_usage=True, 
trust_remote_code=True
    )

    print(f"[merge] Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)

    # IMPORTANT: merge LoRA into base weights and drop PEFT wrappers
    print("[merge] Merging LoRA → base and unloading PEFT …")
    model = model.merge_and_unload()

    # Save tokenizer (take from base)
    print("[merge] Saving tokenizer …")
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True, 
trust_remote_code=True)
    tok.save_pretrained(out_dir)

    # Save merged model
    print(f"[merge] Saving merged model → {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    # Small sanity print
    print("[merge] Done. Files in output directory:")
    for p in sorted(out_dir.glob("*")):
        print("  -", p.name)

if __name__ == "__main__":
    main()

