"""QLoRA fine-tuning of Llama 3.2 3B Instruct into Aura (Unsloth, fits in 6 GB VRAM).

    python -m aura_ml.data.sft_dataset
    python -m aura_ml.training.train_llm --epochs 1
    python -m aura_ml.training.train_llm --eval-only     # re-run evaluation on a saved adapter

Loss is computed only on the final assistant reply of each example: earlier
assistant turns in the history were written under a different context and
strategy, so learning from them would blur the conditioning.

Evaluation (test split): perplexity of the gold replies under the base model vs
the fine-tuned model (same weights, adapter switched off/on), plus ROUGE-L and
distinct-n of sampled generations, and a qualitative samples file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch

# Unsloth must be imported before transformers/peft to apply its patches.
import unsloth  # noqa: F401, I001
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from unsloth import FastLanguageModel

from aura_ml.config import HF_LLM_BASE, LLM_ADAPTER_DIR, SEED, SFT_DATA_DIR
from aura_ml.eval.metrics import distinct_n, rouge_l
from aura_ml.io import read_jsonl, write_json
from aura_ml.training.common import Timer, environment, seed_everything

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize(example: dict, tokenizer, max_len: int) -> dict | None:
    prompt_ids = tokenizer.apply_chat_template(
        example["messages"][:-1], add_generation_prompt=True, tokenize=True
    )
    full_ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
    if len(full_ids) > max_len:  # drop the oldest history turns until it fits
        msgs = example["messages"]
        while len(full_ids) > max_len and len(msgs) > 3:
            msgs = [msgs[0], *msgs[3:]]  # remove one user+assistant pair after the system prompt
            prompt_ids = tokenizer.apply_chat_template(
                msgs[:-1], add_generation_prompt=True, tokenize=True
            )
            full_ids = tokenizer.apply_chat_template(msgs, tokenize=True)
        if len(full_ids) > max_len:
            return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def encode_split(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    out = [t for r in rows if (t := tokenize(r, tokenizer, max_len))]
    print(f"encoded {len(out)}/{len(rows)} examples (max_len={max_len})")
    return out


@torch.inference_mode()
def reply_perplexity(model, tokenizer, items: list[dict], use_adapter: bool, bs: int = 8) -> float:
    collate = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    ctx = model.disable_adapter() if not use_adapter else nullcontext()
    nll, count = 0.0, 0
    model.eval()
    with ctx:
        for i in range(0, len(items), bs):
            batch = {k: v.to(model.device) for k, v in collate(items[i : i + bs]).items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.float()
            shift_logits, shift_labels = logits[:, :-1], batch["labels"][:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            nll += loss.item()
            count += (shift_labels != -100).sum().item()
    return math.exp(nll / count)


@torch.inference_mode()
def generate(
    model, tokenizer, messages: list[dict], use_adapter: bool, max_new_tokens: int = 96
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    ctx = model.disable_adapter() if not use_adapter else nullcontext()
    with ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def evaluate(model, tokenizer, test_rows: list[dict], max_len: int, n_ppl: int, n_gen: int) -> dict:
    rng = random.Random(SEED)
    FastLanguageModel.for_inference(model)
    ppl_items = encode_split(rng.sample(test_rows, min(n_ppl, len(test_rows))), tokenizer, max_len)
    report = {
        "test_reply_perplexity": {
            "base_llama_3.2_3b_instruct": reply_perplexity(
                model, tokenizer, ppl_items, use_adapter=False
            ),
            "aura_finetuned": reply_perplexity(model, tokenizer, ppl_items, use_adapter=True),
            "n": len(ppl_items),
        }
    }
    gen_rows = rng.sample(test_rows, min(n_gen, len(test_rows)))
    samples, outputs = [], {"base": [], "aura": []}
    for row in gen_rows:
        prompt, ref = row["messages"][:-1], row["messages"][-1]["content"]
        base = generate(model, tokenizer, prompt, use_adapter=False)
        aura = generate(model, tokenizer, prompt, use_adapter=True)
        outputs["base"].append(base)
        outputs["aura"].append(aura)
        samples.append(
            {
                "id": row["id"],
                "strategy": row["strategy"],
                "user": prompt[-1]["content"],
                "reference": ref,
                "base": base,
                "aura": aura,
            }
        )
    refs = [s["reference"] for s in samples]
    report["generation"] = {
        name: {
            "rouge_l": sum(rouge_l(p, r) for p, r in zip(preds, refs, strict=True)) / len(refs),
            "distinct_1": distinct_n(preds, 1),
            "distinct_2": distinct_n(preds, 2),
            "avg_words": sum(len(p.split()) for p in preds) / len(preds),
        }
        for name, preds in outputs.items()
    } | {"reference_avg_words": sum(len(r.split()) for r in refs) / len(refs), "n": len(refs)}
    return report, samples


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=SFT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=LLM_ADAPTER_DIR)
    ap.add_argument("--base", default=HF_LLM_BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--n-ppl", type=int, default=400)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    seed_everything(SEED)

    if args.eval_only:
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(args.out_dir), max_seq_length=args.max_len, load_in_4bit=True
        )
        train_minutes = None
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base, max_seq_length=args.max_len, load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        train = encode_split(read_jsonl(args.data_dir / "train.jsonl"), tokenizer, args.max_len)
        val = encode_split(
            random.Random(SEED).sample(read_jsonl(args.data_dir / "validation.jsonl"), 500),
            tokenizer,
            args.max_len,
        )
        bf16 = torch.cuda.is_bf16_supported()
        trainer = Trainer(
            model=model,
            train_dataset=train,
            eval_dataset=val,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
            args=TrainingArguments(
                output_dir=str(args.out_dir / "checkpoints"),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                weight_decay=0.01,
                optim="adamw_8bit",
                bf16=bf16,
                fp16=not bf16,
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                group_by_length=True,
                report_to="none",
                seed=SEED,
            ),
        )
        with Timer() as timer:
            trainer.train()
        train_minutes = round(timer.seconds / 60, 1)
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        write_json(args.out_dir / "train_log.json", {"log_history": trainer.state.log_history})

    report, samples = evaluate(
        model,
        tokenizer,
        read_jsonl(args.data_dir / "test.jsonl"),
        args.max_len,
        args.n_ppl,
        args.n_gen,
    )
    report |= {
        "model": f"{args.base} + LoRA (r={args.rank}, alpha={args.alpha}, all linear layers), 4-bit QLoRA via Unsloth",
        "data": "ESConv hyper-contextual SFT set (aura_ml.data.sft_dataset), loss on final reply only",
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_minutes": train_minutes,
        "environment": environment(),
    }
    write_json(args.out_dir / "metrics.json", report)
    with (args.out_dir / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("test_reply_perplexity", "generation")}, indent=2))


if __name__ == "__main__":
    main()
