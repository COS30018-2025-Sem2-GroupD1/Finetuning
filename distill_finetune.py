#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
distill_finetune.py

Fine-tune a LLaMA-family model on a single distilled JSONL with optional soft-label KD.
- Splits train/val/test in memory (defaults 0.90 / 0.05 / 0.05) with seed.
- Supports LoRA or QLoRA.
- Can resume from an existing LoRA adapter directory to continue training.
- Optionally consumes soft labels (teacher top-k) WHEN tokenizers are compatible.
  Given your teacher (MedGemma) != student (LLaMA), KD is disabled by default;
  you can force-enable with --force-kd at your own risk.

Input JSONL rows must include: instruction, input, output, (optional) id, source, task
Optional embedded soft labels per row:
  "soft_labels": {"topk": K, "steps":[{"t":0,"chosen_id":...,"topk_ids":[...],"topk_logprobs":[...]}, ...]}
Optional external soft labels JSONL(.gz): {"id":"...", "topk":K, "steps":[...]} one per line.

Example (resume from previous LoRA checkpoint and QLoRA):
  python scripts/distill_finetune.py \
    --model-dir model/medalpaca-7b \
    --data-json data/healthcaremagic_distillation.jsonl \
    --out-dir checkpoints/medalpaca_pubmedqa_lora_v2 \
    --use-qlora --bf16 --gradient-checkpointing \
    --resume-adapter-dir checkpoints/medalpaca_pubmedqa_lora/checkpoint-2000 \
    --epochs 2 --batch-size 1 --grad-accum 16 \
    --eval-steps 200 --save-steps 200 --logging-steps 50 \
    --save-splits
"""

import os, json, argparse, math, re, gzip, random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import transformers, peft

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import matplotlib.pyplot as plt
import logging

# --------------------
# I/O helpers
# --------------------
def read_jsonl_any(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = str(path)
    opener = gzip.open if p.endswith(".gz") else open
    rows = []
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows

def load_distilled_rows(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if not all(k in obj for k in ["instruction","input","output"]):
                continue
            # ensure id
            if "id" not in obj:
                obj["id"] = str(len(rows))
            rows.append(obj)
    return rows

def save_jsonl(rows: List[Dict[str, Any]], path: Union[str, Path]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --------------------
# Prompt
# --------------------
def build_prompt(instruction: str, context: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{context.strip()}\n\n"
        "### Response:\n"
    )

# --------------------
# Soft-label utilities
# --------------------
def extract_embedded_soft(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "soft_labels" in rec and isinstance(rec["soft_labels"], dict):
        return rec["soft_labels"]
    meta = rec.get("meta") or {}
    sl = meta.get("soft_labels")
    return sl if isinstance(sl, dict) else None

def load_soft_map(soft_file: Optional[str]) -> Dict[str, Any]:
    if not soft_file:
        return {}
    mp: Dict[str, Any] = {}
    for rec in read_jsonl_any(soft_file):
        rid = str(rec.get("id"))
        if rid:
            mp[rid] = rec
    return mp

def _soft_steps_ok(soft: Dict[str, Any]) -> bool:
    return isinstance(soft, dict) and isinstance(soft.get("steps"), list) and soft.get("steps")

def _steps_as_probs(soft: Dict[str, Any], eps: float = 1e-8) -> List[Tuple[List[int], List[float]]]:
    out = []
    for st in soft.get("steps", []):
        ids = st.get("topk_ids") or []
        lps = st.get("topk_logprobs") or []
        if not ids or not lps:
            out.append(([], [])); continue
        ps = np.exp(np.array(lps, dtype=np.float64))
        s = float(ps.sum()) + eps
        ps = (ps / s).tolist()
        out.append((ids, ps))
    return out

# --------------------
# Dataset
# --------------------
@dataclass
class DistillDataset(Dataset):
    data: List[Dict[str, Any]]
    tokenizer: Any
    soft_map: Dict[str, Any]
    use_soft: bool
    kd_temp: float
    max_len: int = 2048

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        rid = str(ex.get("id", idx))

        prompt = build_prompt(ex["instruction"], ex["input"])
        answer = (ex["output"] or "").strip()
        if self.tokenizer.eos_token and not answer.endswith(self.tokenizer.eos_token):
            answer = answer + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:self.max_len]
        used_prompt_len = min(len(prompt_ids), len(input_ids))

        labels = [-100]*used_prompt_len
        remain = self.max_len - len(labels)
        labels += answer_ids[:max(0, remain)]
        labels = labels[:len(input_ids)]

        attn = [1]*len(input_ids)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": used_prompt_len,
            "id": rid,
        }

        soft = None
        if self.use_soft:
            soft = extract_embedded_soft(ex)
            if soft is None and rid in self.soft_map:
                soft = self.soft_map[rid]
        item["soft"] = soft
        return item

def pad_to_max(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "prompt_len": [], "id": [], "soft": []}
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        out["labels"].append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        out["prompt_len"].append(x["prompt_len"])
        out["id"].append(x["id"])
        out["soft"].append(x["soft"])
    return {
        "input_ids": torch.stack(out["input_ids"], dim=0),
        "attention_mask": torch.stack(out["attention_mask"], dim=0),
        "labels": torch.stack(out["labels"], dim=0),
        "prompt_len": torch.tensor(out["prompt_len"], dtype=torch.long),
        "id": out["id"],
        "soft": out["soft"],
    }

# --------------------
# KD loss over subset (top-k)
# --------------------
def kd_loss_from_subset_logits(
    logits: torch.Tensor, labels: torch.Tensor, prompt_len: torch.Tensor,
    soft_batch: List[Optional[Dict[str, Any]]], temperature: float = 1.0,
) -> torch.Tensor:
    B, T, V = logits.shape
    device = logits.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for b in range(B):
        soft = soft_batch[b]
        if not _soft_steps_ok(soft):
            continue
        pairs = _steps_as_probs(soft)
        start = int(prompt_len[b].item())
        max_j = min(len(pairs), T - start)
        if max_j <= 0:
            continue
        for j in range(max_j):
            ids, probs = pairs[j]
            if not ids:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            t_probs = torch.tensor(probs, dtype=torch.float32, device=device)

            s_logits_sel = logits[b, start + j, ids_t] / max(1e-6, temperature)
            s_logsumexp = torch.logsumexp(s_logits_sel, dim=-1, keepdim=True)
            s_log_probs_sel = s_logits_sel - s_logsumexp

            t_log_probs = torch.log(t_probs + 1e-8)
            kl = torch.sum(t_probs * (t_log_probs - s_log_probs_sel))
            total = total + kl
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count

# --------------------
# Minimal text eval helpers
# --------------------
@torch.no_grad()
def generate_batch_text(model, tok, prompts: List[str], gen_kw: Dict[str, Any]) -> List[str]:
    outs = []
    for pr in prompts:
        enc = tok(pr, return_tensors="pt").to(model.device)
        out = model.generate(**enc, **gen_kw)
        text = tok.decode(out[0], skip_special_tokens=True)
        prompt_text = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
        outs.append(text[len(prompt_text):].strip())
    return outs

def simple_exact_ynm(preds: List[str], refs: List[str]) -> float:
    def norm_first(s: str) -> str:
        if not s: return ""
        s = s.strip().lower()
        m = re.match(r"[a-z]+", s)
        return m.group(0) if m else s.split()[0] if s.split() else ""
    ok = 0
    for p, r in zip(preds, refs):
        ok += int(norm_first(p) == norm_first(r))
    return ok / max(1, len(preds))

# --------------------
# Viz + logger
# --------------------
def visualize_training_progress(losses, val_losses: List[float]) -> None:
    try:
        plt.figure(figsize=(10,5))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Steps"); plt.ylabel("Loss"); plt.legend()
        plt.savefig("training_progress.png")
        print("Training progress saved to training_progress.png")
    except Exception as e:
        print(f"Warning: Could not generate training progress plot: {e}")

def setup_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.txt")
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler(); console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    return logging.getLogger(__name__)

# --------------------
# KD-enabled Trainer
# --------------------
from transformers import Trainer
class KDTrainer(Trainer):
    def __init__(self, *args, kd_weight: float = 0.5, kd_temperature: float = 1.0, use_soft_labels: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.use_soft_labels = use_soft_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"])
        ce_loss = outputs.loss
        if not self.use_soft_labels:
            return (ce_loss, outputs) if return_outputs else ce_loss
        kd = kd_loss_from_subset_logits(
            outputs.logits, inputs["labels"], inputs["prompt_len"], inputs["soft"], temperature=self.kd_temperature
        )
        loss = ce_loss + (self.kd_weight * kd)
        return (loss, outputs) if return_outputs else loss

# --------------------
# Split util
# --------------------
def train_val_test_split(rows: List[Dict[str, Any]],
                         train_ratio: float, val_ratio: float, test_ratio: float,
                         seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train+n_val]
    test_idx  = idxs[n_train+n_val:]
    train = [rows[i] for i in train_idx]
    val   = [rows[i] for i in val_idx]
    test  = [rows[i] for i in test_idx]
    return train, val, test

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Base LLaMA-7B dir (HF format)")
    ap.add_argument("--data-json", required=True, help="Single distilled JSONL to be split")
    ap.add_argument("--out-dir", required=True, help="Checkpoint output directory")

    # Splits
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio",   type=float, default=0.05)
    ap.add_argument("--test-ratio",  type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-splits", action="store_true", help="Write split JSONLs under out-dir/splits/")

    # Soft labels
    ap.add_argument("--use-soft-labels", action="store_true", help="Enable KD if compatible or forced.")
    ap.add_argument("--soft-file", default=None, help="Optional JSONL(.gz) with soft labels (id->steps) for the WHOLE dataset.")
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temperature", type=float, default=1.0)
    ap.add_argument("--force-kd", action="store_true", help="Force-enable KD even if tokenizer mismatch (use with caution).")

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", choices=["linear","cosine"], default="linear")
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=2048)

    # LoRA / QLoRA
    ap.add_argument("--use-qlora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # Precision & misc
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")

    # Early stopping
    ap.add_argument("--early-stopping-patience", type=int, default=3)

    # Sanity generation on val
    ap.add_argument("--gen-sample-size", type=int, default=128)

    # Resume options
    ap.add_argument("--resume-adapter-dir", default=None,
                    help="Path to existing LoRA adapter (e.g., checkpoints/medalpaca_pubmedqa_lora/checkpoint-2000) to continue training from those weights.")
    ap.add_argument("--resume-trainer-dir", default=None,
                    help="Optional Trainer checkpoint dir (to resume optimizer/scheduler state).")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir)
    logger.info("===== Training run started =====")
    logger.info(f"Args: {args}")
    logger.info(f"Versions — transformers: {transformers.__version__}, peft: {peft.__version__}, torch: {torch.__version__}")

    # Load data and split
    all_rows = load_distilled_rows(args.data_json)
    train_rows, val_rows, test_rows = train_val_test_split(
        all_rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed
    )
    logger.info(f"Split sizes => train: {len(train_rows)}  val: {len(val_rows)}  test: {len(test_rows)}")

    if args.save_splits:
        split_dir = Path(args.out_dir) / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(train_rows, split_dir / "train.jsonl")
        save_jsonl(val_rows,   split_dir / "val.jsonl")
        save_jsonl(test_rows,  split_dir / "test.jsonl")
        logger.info(f"Saved splits to {split_dir}")

    # Tokenizer / Model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    load_kwargs = {}
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        load_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    base_model.config.pad_token_id = tok.pad_token_id
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    if args.use_qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    # Attach LoRA: resume if provided, else create new
    #if args.resume-adapter-dir:
    #    # (hyphen is invalid in attribute; fix argparse variable)
    #    pass

    # ---------- Resume adapter or new LoRA ----------
    resume_adapter_dir = args.resume_adapter_dir
    if resume_adapter_dir:
        logger.info(f"Loading existing LoRA adapter from: {resume_adapter_dir}")
        model = PeftModel.from_pretrained(base_model, resume_adapter_dir, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=args.target_modules
        )
        model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ---------- Soft labels map (global) ----------
    soft_map_all = load_soft_map(args.soft_file) if args.soft_file else {}

    # KD default OFF unless user forces it (since teacher != student tokenizer in your setup)
    use_kd = bool(args.use_soft_labels and (args.force_kd))
    if args.use_soft_labels and not args.force_kd:
        logger.warning("KD requested but not forced. Teacher/student tokenizers likely mismatch (Gemma vs LLaMA). "
                       "Proceeding with HARD-label SFT. Use --force-kd to override at your own risk.")

    # ---------- Build datasets ----------
    train_ds = DistillDataset(train_rows, tok, soft_map_all, use_kd, args.kd_temperature, max_len=args.max_len)
    val_ds   = DistillDataset(val_rows,   tok, soft_map_all, use_kd, args.kd_temperature, max_len=args.max_len)
    test_ds  = DistillDataset(test_rows,  tok, soft_map_all, use_kd, args.kd_temperature, max_len=args.max_len)

    def collate(batch):
        return pad_to_max(batch, tok.pad_token_id)

    # ---------- Training args ----------
    def _has_field(cls, name: str) -> bool:
        try:
            return name in getattr(cls, "__dataclass_fields__", {})
        except Exception:
            return False

    # Map your CLI scheduler to whatever this transformers build expects
    def _scheduler_kwargs(scheduler: str):
        kw = {}
        if _has_field(TrainingArguments, "lr_scheduler_type"):
            kw["lr_scheduler_type"] = scheduler
        elif _has_field(TrainingArguments, "lr_scheduler"):
            kw["lr_scheduler"] = scheduler
        return kw

    # Base kwargs that are almost always present
    ta_kwargs = {
        "output_dir": args.out_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
    }

    # Optional: warmup
    if _has_field(TrainingArguments, "warmup_ratio"):
        ta_kwargs["warmup_ratio"] = args.warmup_ratio
    elif _has_field(TrainingArguments, "warmup_steps"):
        # Conservative fallback (can tune later)
        # If you want exact steps, compute after you know total steps.
        ta_kwargs["warmup_steps"] = 0

    # Precision flags
    if _has_field(TrainingArguments, "bf16"):
        ta_kwargs["bf16"] = args.bf16
    if _has_field(TrainingArguments, "fp16"):
        ta_kwargs["fp16"] = (args.fp16 and not args.bf16)

    # Reporting / misc
    if _has_field(TrainingArguments, "report_to"):
        ta_kwargs["report_to"] = "none"
    if _has_field(TrainingArguments, "dataloader_pin_memory"):
        ta_kwargs["dataloader_pin_memory"] = False
    if _has_field(TrainingArguments, "save_total_limit"):
        ta_kwargs["save_total_limit"] = 2

    # Evaluation controls (version-adaptive)
    eval_enabled = False
    if _has_field(TrainingArguments, "evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = "steps"
        ta_kwargs["eval_steps"] = args.eval_steps
        if _has_field(TrainingArguments, "load_best_model_at_end"):
            ta_kwargs["load_best_model_at_end"] = True
        if _has_field(TrainingArguments, "metric_for_best_model"):
            ta_kwargs["metric_for_best_model"] = "eval_loss"
        if _has_field(TrainingArguments, "greater_is_better"):
            ta_kwargs["greater_is_better"] = False
        eval_enabled = True
    elif _has_field(TrainingArguments, "evaluate_during_training"):
        # Very old Transformers (pre 3.x)
        ta_kwargs["evaluate_during_training"] = True
        # Older builds typically also support eval_steps
        if _has_field(TrainingArguments, "eval_steps"):
            ta_kwargs["eval_steps"] = args.eval_steps
        eval_enabled = True
    # else: no evaluation support in this build

    # Scheduler
    ta_kwargs.update(_scheduler_kwargs(args.scheduler))

    # Construct TrainingArguments with the filtered kwargs
    train_args = TrainingArguments(**ta_kwargs)

    # Early stopping only if evaluation is actually wired
    callbacks = []
    if eval_enabled and _has_field(TrainingArguments, "evaluation_strategy"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    trainer = KDTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if eval_enabled else None,
        data_collator=collate,
        callbacks=callbacks,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        use_soft_labels=use_kd,
    )

    # ---------- Train ----------
    train_out = trainer.train(resume_from_checkpoint=args.resume_trainer_dir)
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    # ---------- Small gen sanity on val ----------
    gen_n = min(args.gen_sample_size, len(val_rows))
    if gen_n > 0:
        logger.info(f"Running small-gen sanity check on {gen_n} samples...")
        gen_prompts = [build_prompt(r["instruction"], r["input"]) for r in val_rows[:gen_n]]
        gen_refs    = [r["output"] for r in val_rows[:gen_n]]
        gen_kw = dict(max_new_tokens=128, do_sample=False, temperature=0.0,
                      repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        preds = generate_batch_text(trainer.model, tok, gen_prompts, gen_kw)
        ynm = simple_exact_ynm(preds, gen_refs)
        logger.info(f"[VAL small-gen] Y/N/Maybe exact-ish: {ynm:.3f}")

    # ---------- Eval ----------
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")
    test_metrics = trainer.evaluate(test_ds)
    logger.info(f"Test metrics: {test_metrics}")

    # ---------- Save ----------
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # Save logs
    log_json = Path(args.out_dir) / "training_log.json"
    try:
        with open(log_json, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)
        logger.info(f"Training logs saved to: {log_json}")
    except Exception as e:
        logger.warning(f"Could not save training logs: {e}")

    # Viz
    try:
        losses = [e['loss'] for e in trainer.state.log_history if 'loss' in e]
        vloss  = [e['eval_loss'] for e in trainer.state.log_history if 'eval_loss' in e]
        visualize_training_progress(losses, vloss)
    except Exception as e:
        logger.warning(f"viz error: {e}")

    logger.info("===== Training run completed =====")

if __name__ == "__main__":
    main()
