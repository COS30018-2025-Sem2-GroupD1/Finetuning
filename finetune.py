#!/usr/bin/env python
"""
Fine-tune X model on y.jsonl (train/val/test).

Input JSONL fields (one per line):
  instruction, input, output, source, id, task

Usage (LoRA):
  python scripts/finetune.py \
    --model-dir model/medalpaca-7b \
    --train-json data/processed/pubmedqa_train.jsonl \
    --val-json   data/processed/pubmedqa_val.jsonl \
    --test-json  data/processed/pubmedqa_test.jsonl \
    --out-dir    checkpoints/medalpaca_pubmedqa_lora

Add --use-qlora to use 4-bit QLoRA (requires bitsandbytes wheel that matches the node).
"""

import os, json, argparse, math, re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import json 

# --------------------
# Data
# --------------------
FIELDS = ["instruction","input","output","source","id","task"]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # basic validation
            if not all(k in obj for k in ["instruction","input","output"]):
                continue
            rows.append(obj)
    return rows

def build_prompt(instruction: str, context: str) -> str:
    # Simple Alpaca-style prompt
    return (
	"### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{context.strip()}\n\n"
        "### Response:\n"
    )

@dataclass
class PubMedQADataset(Dataset):
    data: List[Dict[str, Any]]
    tokenizer: Any
    max_len: int = 2048

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = build_prompt(ex["instruction"], ex["input"])
        answer = ex["output"].strip()
        if not answer.endswith(self.tokenizer.eos_token or ""):
            answer = answer + (self.tokenizer.eos_token or "")

        # Tokenize separately to mask prompt in labels
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        # Truncate if too long
        input_ids = (prompt_ids + answer_ids)[:self.max_len]
        # Label mask: -100 for prompt, real ids for answer (clipped to length)
        labels = [-100]*min(len(prompt_ids), len(input_ids))
        labels += answer_ids[: max(0, self.max_len - len(labels))]
        labels = labels[:len(input_ids)]

        attn = [1]*len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def pad_to_max(batch, pad_id: int):
    # manual pad because we need to pad labels with -100
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        # labels pad with -100
        out["labels"].append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    out = {k: torch.stack(v, dim=0) for k, v in out.items()}
    return out

# --------------------
# Metrics
# --------------------
def simple_exact_ynm(preds: List[str], refs: List[str]) -> float:
    """
    Light exact match for yes/no/maybe tasks.
    Normalizes to first token of each answer and compares.
    If the dataset has long rationales, this will be superficial;
    we still rely on eval_loss for early stopping.
    """
    def norm_first(s: str) -> str:
        if not s: return ""
        s = s.strip().lower()
        # take first word-ish token
        m = re.match(r"[a-z]+", s)
        return m.group(0) if m else s.split()[0] if s.split() else ""
    ok = 0
    for p, r in zip(preds, refs):
        ok += int(norm_first(p) == norm_first(r))
    return ok / max(1, len(preds))

# --------------------
# Generation helper for small eval set
# --------------------
@torch.no_grad()
def generate_batch_text(model, tok, prompts: List[str], gen_kw: Dict[str, Any]) -> List[str]:
    outs = []
    for pr in prompts:
        enc = tok(pr, return_tensors="pt").to(model.device)
        out = model.generate(**enc, **gen_kw)
        text = tok.decode(out[0], skip_special_tokens=True)
        # strip the prompt to get the model's response part
        prompt_text = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
        resp = text[len(prompt_text):].strip()
        outs.append(resp)
    return outs


###### Visualization
def visualize_training_progress(losses, val_losses: List[float]) -> None:
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_progress.png")
        print("Training progress saved to training_progress.png")
    except Exception as e:
        print(f"Warning: Could not generate training progress plot: {e}")
    
# --------------------
# Logger
# --------------------
import logging
def setup_logger(out_dir: str):
    log_path = os.path.join(out_dir, "training_log.txt")
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Also stream to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    return logging.getLogger(__name__)


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Local path to medalpaca-7b")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json",   required=True)
    ap.add_argument("--test-json",  required=True)
    ap.add_argument("--out-dir",    required=True)

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)  # increase if VRAM allows
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", choices=["linear","cosine"], default="linear")
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA / QLoRA
    ap.add_argument("--use-qlora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # precision & misc
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")

    # early stopping
    ap.add_argument("--early-stopping-patience", type=int, default=3)

    # small generation eval
    ap.add_argument("--gen-sample-size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Logger
    logger = setup_logger(args.out_dir)
    logger.info("===== Training run started =====")
    logger.info(f"Args: {args}")

    # Seed
    torch.manual_seed(args.seed)

    # Tokenizer / Model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    load_kwargs = {}
    if args.use_qlora:
        # 4-bit QLoRA path (requires bitsandbytes)
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["device_map"] = "auto"
    else:
        # standard FP16/BF16 (LoRA)
        load_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    model.config.pad_token_id = tok.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    train_rows = load_jsonl(args.train_json)
    val_rows   = load_jsonl(args.val_json)
    test_rows  = load_jsonl(args.test_json)

    train_ds = PubMedQADataset(train_rows, tok, max_len=args.max_len)
    val_ds   = PubMedQADataset(val_rows, tok, max_len=args.max_len)
    test_ds  = PubMedQADataset(test_rows, tok, max_len=args.max_len)

    def collate(batch):
        return pad_to_max(batch, tok.pad_token_id)

    # Training args
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to="none",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    callbacks = []
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        callbacks=callbacks,
    )

    # Train
    train_out = trainer.train()
    logger.info("Best checkpoint:", trainer.state.best_model_checkpoint)

    # Quick generation-based sanity eval on a small sample of val
    gen_n = min(args.gen_sample_size, len(val_rows))
    logger.info(f"Running small-gen sanity check on {gen_n} samples...")
    gen_prompts = [build_prompt(r["instruction"], r["input"]) for r in val_rows[:gen_n]]
    gen_refs    = [r["output"] for r in val_rows[:gen_n]]

    gen_kw = dict(
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0,
        pad_token_id=tok.pad_token_id,
    )
    preds = generate_batch_text(trainer.model, tok, gen_prompts, gen_kw)
    ynm_acc = simple_exact_ynm(preds, gen_refs)
    logger.info(f"[VAL small-gen] Y/N/Maybe exact-ish: {ynm_acc:.3f}")

    # Final eval loss on full val
    eval_metrics = trainer.evaluate()
    logger.info("Eval metrics:", eval_metrics)

    # Test loss
    test_metrics = trainer.evaluate(test_ds)
    logger.info("Test metrics:", test_metrics)

    # Save last
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    
    # Save training logs 
    log_path = os.path.join(args.out_dir, "training_log.json")
    try:
        with open(log_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)
        print(f"Training logs saved to: {log_path}")
    except Exception as e:
        print(f"Warning: Could not save training logs: {e}")

    # Visualization
    visualize_training_progress(
    [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry],
    [m['eval_loss'] for m in trainer.state.log_history if 'eval_loss' in m]
    )
    logger.info("===== Training run completed =====")


if __name__ == "__main__":
    main()

