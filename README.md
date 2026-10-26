# Complementary Fashion Item Retrieval on Polyvore Outfits

> Independent mini-project (Oct 2026), exploring complementary item retrieval for fashion recommendation.

Given a partial outfit, retrieve the missing item from **every** test item of the target category (1.4K–13K candidates), not from a 4-way multiple choice.

**TL;DR**
- A small model trained on frozen CLIP embeddings improves Recall@10 by **2.3×** over zero-shot CLIP (3.66 → 8.25).
- In a parameter-matched comparison (~1.6M params, 3 seeds), a simple **additive** set model outperforms a Transformer set encoder.
- The apparent advantage of attention in early runs came from an **input-scale mismatch** in the baselines, not from the architecture.
- The additive model is also the cheaper one to serve: per-item query vectors can be cached and pooled, and the result goes straight into an ANN index.

## Task

- **Data:** Polyvore Outfits, disjoint split. 152,785 unique items, 11 categories.
- **Query:** an outfit with one item removed. The model is told the target category.
- **Candidates:** all test items of that category. Items already in the query are excluded.
- **Metrics:** Recall@10, Recall@50, MRR over 74,000 test queries.

**Leakage control.** The "disjoint" split still shares some items across splits (train∩valid 3,781, train∩test 84, valid∩test 34). Test outfits that contain *any* train or valid item are dropped (49 outfits). Validation items that also appear in train are never used as targets.

## Method

Images are embedded once with OpenCLIP ViT-B/32 (LAION-2B) and kept frozen. All learned models share the candidate tower and a learned category embedding, and differ only in how they encode the query set:

| Model | Query encoder |
|---|---|
| Learned additive | MLP on each (item, target category) pair, then mean. No item–item interaction. |
| DeepSets | Same, plus a nonlinear transform after pooling. |
| Set encoder | Category token + items through a 2-layer Transformer; read out the category token. |

Training uses an in-batch contrastive loss where every batch shares one target category, so in-batch negatives are same-category items, which matches the evaluation. Duplicate targets and targets that appear in another query's context are masked. Query items are randomly subsampled during training for robustness to set size.

## Results

Test set, mean ± std over 3 seeds. Learned models are matched at ~1.6M parameters.

| Method | R@10 | R@50 | MRR |
|---|---|---|---|
| Random (expected) | 0.16 | 0.79 | 0.14 |
| Zero-shot: per-item max | 2.80 | 7.39 | 1.58 |
| Zero-shot: CLIP mean | 3.66 | 9.57 | 1.94 |
| Learned additive | **8.25 ± 0.02** | 20.65 ± 0.05 | **4.10 ± 0.07** |
| DeepSets | 8.21 ± 0.08 | **20.66 ± 0.16** | 4.08 ± 0.04 |
| Set encoder (attention) | 7.90 ± 0.05 | 20.09 ± 0.18 | 3.90 ± 0.05 |

### The attention advantage was an optimization artifact

In the first parameter-matched runs, the set encoder won by a wide margin. The MLP baselines' training loss sat near ln(512) ≈ 6.24, the chance level for a batch of 512, for several epochs. The cause was concatenating unit-norm CLIP vectors with a much larger category embedding. Adding LayerNorm doubled the additive model's score and reversed the ranking:

| Model | Without LayerNorm | With LayerNorm |
|---|---|---|
| Learned additive | 4.03 ± 0.14 | **8.25 ± 0.02** |
| DeepSets | 2.71 ± 0.70 | 8.21 ± 0.08 |
| Set encoder | 7.72 ± 0.03 | 7.90 ± 0.05 |

The MLP models get LayerNorm on inputs and hidden layers. The set encoder already uses pre-norm Transformer layers, so it only gets input LayerNorm.

### Attention does not help even when items are diverse (exploratory)

Hypothesis: when query items differ a lot, modeling their interactions should matter more. Queries with ≥2 items were split into thirds by mean pairwise CLIP similarity.

| R@10 | Diverse | Medium | Similar |
|---|---|---|---|
| Learned additive | 7.54 | 7.96 | 9.36 |
| Set encoder | 7.18 | 7.73 | 8.90 |

The additive model leads in every bin.

### More context helps, once the query is held fixed

Aggregated R@10 peaks at 3 query items and then drops. Reweighting to a fixed category mix does not remove this. To separate "more items" from "harder outfits", queries with ≥5 items were re-evaluated while revealing only the first *k* items (33,420 queries):

| R@10 | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| CLIP mean | 1.84 | 2.52 | 2.82 | 2.99 | 3.19 |
| Learned additive | 3.05 | 4.83 | 6.06 | 7.05 | 7.85 |
| Set encoder | 2.74 | 4.60 | 5.84 | 6.67 | 7.47 |

All methods improve monotonically, so the dip comes from larger outfits being harder, not from extra context hurting. Learned models gain +4.8 pts from k=1 to 5, while zero-shot gains +1.35.

## Why this matters for serving

The additive model scores a query as a mean of per-item vectors φ(item, category). These vectors can be precomputed and cached. Adding a pin to a board only updates a running mean, and the result is a single vector for ANN retrieval. In this setting, the cheaper architecture is also the more accurate one.

## Limitations

- **Scope.** The conclusion holds for frozen CLIP features at ~1.6M parameters. The set encoder was not tuned separately (same LR, schedule, and epochs as the baselines).
- **Overfitting.** MLP models peak on validation around epoch 13–16 and are selected by best checkpoint. A shorter schedule may be better.
- **Valid vs. test numbers.** Validation R@10 (~22%) is not comparable to test (~8%) because validation candidate pools are about 5× smaller.
- **Image only.** Item titles and descriptions are unused.
- **Benchmarks.** Standard FITB accuracy and compatibility AUC are not reported yet.

## Reproduce

1. Request access to the gated dataset [`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits) on Hugging Face.
2. In a Colab notebook with a T4 GPU:

   ```
   !git clone https://github.com/lee-research/outfit-completion-retrieval.git
   %cd outfit-completion-retrieval
   !pip -q install open_clip_torch huggingface_hub
   %run run_experiments.py
   ```

Embeddings and checkpoints are cached to Google Drive, so later runs skip extraction and training.

```
├── README.md
├── run_experiments.py
├── requirements.txt
└── .gitignore
```

## Data

Polyvore Outfits ([Vasileva et al., ECCV 2018](https://arxiv.org/abs/1803.09196)) is restricted to academic, non-profit research use. The dataset authors do not own the image copyrights. This repository contains **no images, metadata, or embeddings**; it only contains code that downloads the data after access is granted.
