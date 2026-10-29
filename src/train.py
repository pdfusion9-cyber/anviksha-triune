"""Train either the baseline or the triune model on the same data and budget.

    python src/train.py --model baseline --steps 300
    python src/train.py --model triune   --steps 300

Both runs use identical data order, identical batch size and identical step
count, so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CharDataset  # noqa: E402
from model import BaselineLM, TriuneLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "triune"], required=True)
    parser.add_argument("--data", default="../hindi_corpus.txt", type=Path)
    parser.add_argument("--out", default="runs", type=Path)
    parser.add_argument("--dim", default=96, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--seq-len", default=128, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--max-chars", default=1_500_000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    dataset = CharDataset(args.data, args.seq_len, args.max_chars)
    print(f"vocab {dataset.vocab_size}  train chars {len(dataset.train):,}  "
          f"valid chars {len(dataset.valid):,}")

    common = dict(vocab=dataset.vocab_size, dim=args.dim, layers=args.layers,
                  heads=args.heads, seq_len=args.seq_len)
    model = BaselineLM(**common) if args.model == "baseline" else TriuneLM(**common)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {parameters:,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.time()
    history = []

    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = dataset.batch(dataset.train, args.batch, args.seq_len, rng)
        optimiser.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                valid_inputs, valid_targets = dataset.batch(
                    dataset.valid, args.batch, args.seq_len, np.random.default_rng(1)
                )
                valid_logits = model(valid_inputs)
                valid_loss = F.cross_entropy(
                    valid_logits.reshape(-1, valid_logits.shape[-1]),
                    valid_targets.reshape(-1),
                ).item()
            elapsed = time.time() - started
            history.append({"step": step, "train_loss": loss.item(),
                            "valid_loss": valid_loss, "seconds": elapsed})
            print(f"step {step:4d}  train {loss.item():.4f}  valid {valid_loss:.4f}  "
                  f"ppl {math.exp(valid_loss):7.2f}  {elapsed:6.0f}s", flush=True)

    metrics = {
        "model": args.model,
        "parameters": parameters,
        "steps": args.steps,
        "seconds": round(time.time() - started, 1),
        "final_valid_loss": history[-1]["valid_loss"],
        "final_perplexity": math.exp(history[-1]["valid_loss"]),
        "history": history,
    }
    if getattr(model, "memory", None) is not None:
        metrics["memory_stats"] = model.memory.stats.as_dict()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"final valid loss {metrics['final_valid_loss']:.4f}  "
          f"perplexity {metrics['final_perplexity']:.2f}")
    if "memory_stats" in metrics:
        print("memory:", metrics["memory_stats"])


if __name__ == "__main__":
    main()
