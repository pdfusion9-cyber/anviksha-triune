# Anviksha Triune — a second hat

An experiment, kept in its own repository. The main project is untouched.

The brief: give the transformer a brain-like memory — conscious layers where
thoughts form and decay and nothing is permanent, a subconscious store that keeps
what proved useful, and an unconscious store for what has aged out. Plus a
2×2×2 bank of smart nodes, so the model spends little compute to get a lot.

This repository implements that, trains it, and measures it against a plain
transformer on the same data and the same step budget. The measurement is the
point, and the result is not flattering — see below.

## What the three tiers actually are

| Tier | Mechanism | Size (default) | On the hot path |
|---|---|---|---|
| conscious | working slots, decayed every step, rewritten per position | 8 slots | yes |
| subconscious | persistent rows, written only by promotion, read by top-k similarity | 128 rows | top-4 only |
| unconscious | cold store, filled when subconscious overflows | 512 rows | no |

Nothing in the conscious tier is permanent. A slot accumulates *vitality* as it
is used, decays by 0.9 per step, and is zeroed when it falls below 0.05. When a
slot crosses 0.55 it is promoted into subconscious, deduplicated against rows
already there by cosine similarity > 0.95. When subconscious is full, its least
used row is demoted to the unconscious tier.

Only conscious is on the hot path. Subconscious costs O(k·D) for k=4, so its
size does not enter per-token cost. That is the "less compute, more capability"
claim, and it is structurally true: the store can grow without the forward pass
getting slower.

## The 2×2×2 node bank

Eight expert nodes per block, routed by a linear router that keeps the top two
per token. The bank holds eight nodes' worth of parameters and costs about two
nodes' worth of compute. This is mixture-of-experts with a small fan-out; the
"2×2×2" is the addressing scheme, not a new mechanism.

## Result — measured, and not what the design hoped for

Hindi character-level language modelling, 1.5 M characters from Hindi Wikipedia.
Same data order, same batch (8), same steps (120), same seed, single thread.

| Model | Parameters | Final valid loss | Perplexity | Time |
|---|---|---|---|---|
| Baseline transformer | 418,848 | 3.3629 | 28.87 | 25.3 s |
| Triune | 1,110,172 | **3.3564** | **28.69** | 42.0 s |

**Quality is a tie (0.2 %), and the triune model costs 2.6× the parameters and
1.7× the time to get it.** At this budget the memory tiers do not pay for
themselves.

That is the honest headline. It is also useful, because the memory stats show
exactly what happened:

```
promotions 412 · demotions 284 · evictions 21,000
subconscious occupancy 100 % · unconscious occupancy 55 %
```

The hierarchy *works* — rows are promoted, deduplicated, and demoted when
subconscious fills. But the store fills within the first few hundred steps, so
most of the run is spent in the overflow path, and the top-k retrieval is
returning rows that were consolidated early and never refreshed. In other words:
the write policy is too eager and the read policy is too static.

## What to fix before drawing a conclusion

1. **120 steps is far too short.** A memory mechanism cannot show value before
   the base model has learned anything to store. This needs 5,000+ steps.
2. **The promotion threshold is untuned.** 0.55 with decay 0.9 promotes almost
   everything within three steps. It should be a function of the model's own
   confidence, not a fixed constant.
3. **Retrieval never updates usage.** Rows that are read should gain vitality;
   currently only promotions and demotions move the counters.
4. **The node bank and the memory are confounded.** Two changes at once means
   neither can be attributed. They should be ablated separately.

## What is real and what is metaphor

Real, and worth keeping:

- A bounded, decaying working memory with an explicit write-back policy.
- A persistent store addressed by similarity, with promotion and eviction.
- Sparse expert routing that decouples parameters from per-token compute.

Metaphor, and should not be sold as more:

- "Conscious" and "unconscious" are names for cache tiers. There is no
  phenomenal experience, no attention schema, no self-model.
- Nothing here demonstrates that the model *thinks*. It demonstrates that a
  cache hierarchy can be bolted onto a transformer without breaking training.
- The brain does not run backpropagation, and this does.

## Run it

```bash
python src/train.py --model baseline --steps 120 --batch 8 --threads 1
python src/train.py --model triune   --steps 120 --batch 8 --threads 1
```

Metrics land in `runs/<model>_metrics.json`, including the tier counters for the
triune run.

## Layout

```
src/memory.py   three-tier store: decay, promotion, dedup, demotion
src/blocks.py   baseline block, 2x2x2 node bank, triune block
src/model.py    BaselineLM and TriuneLM
src/data.py     character-level dataset
src/train.py    identical training loop for both models
```
