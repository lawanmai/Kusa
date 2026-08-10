# kusa — Morphology-Aware Sentiment Analysis for Sorani Kurdish

Code and reproducibility artefacts for a dual-view adaptation of the SARF
architecture to Sorani Kurdish sentiment analysis on the KurdiSent corpus,
evaluated under a leakage-controlled held-out protocol.

A statistical design with seed pairing and minimal HPO. When the expected effect
is below one F1 point, the adversary is seed noise, not the amount of data — so
the compute budget goes into repetitions, not into a large hyperparameter search.

- 20% test set, drawn group-aware and locked away by SHA-256, read by a single
  script at the very end.
- Grouped 5-fold CV on the development pool, with an identical fold assignment
  for all architectures.
- Encoder hyperparameters fixed and shared by every architecture; each distinct
  head runs its **own** exhaustive grid search (24 configurations). Equal search
  budgets, not a shared configuration, are what make the comparison fair.
- Epoch count searched per architecture, no early stopping, collapse guard
  (loss > ln 3 → restart with a different seed).
- The **five fold models** of each architecture score the test set → a spread
  estimate without extra runs, plus paired significance tests over seeds and
  over items.

> **Status:** results are being regenerated after a protocol correction (the
> baseline's classification head now trains in its own parameter group at its own
> searched learning rate, and a capacity-matched control was added). This repository
> currently ships the code and the protocol; result files will be committed once
> the runs complete.

## Architectures

| variant | head | HPO |
|---|---|---|
| `baseline_v2` | XLM-R classification head | own search |
| `dual_view_dupinput_v2` | identical to dual-view, second view = surface | inherits `dual_view_v2` |
| `dual_view_v2` | cross-attention + CNN-BiLSTM | own search |
| `dual_view_gated_v2` | same + gate (probe) | inherits `dual_view_v2` |

`dual_view_dupinput_v2` is the capacity-matched control. It is the dual-view
model itself — same architecture, same 7.84M-parameter head, same configuration —
fed the surface view in place of the lemma view, so its cross-attention
degenerates to self-attention and the second branch carries no information the
first does not. That makes two comparisons possible:

- `baseline_v2` → control: the architectural gain (head plus attention capacity)
- control → `dual_view_v2`: **the morphological information itself**

The second is parameter- and configuration-matched, which a single-view ablation
cannot be — removing the cross-attention would drop 4.2M parameters at the same
time and confound capacity with information. Both controls inherit the dual-view
configuration by necessity: searching separately would give the arms different
hyperparameters and make the comparison two-variable again.

## Hyperparameters

Fixed and identical for every architecture:

| parameter | value | source |
|---|---|---|
| `batch_size` | 32 | Devlin et al. (2019), App. A.3 |
| `weight_decay` | 0.01 | Liu et al. (2019) |
| `warmup_ratio` | 0.10 | Liu et al. (2019) |
| `encoder_lr` | 1e-5 | The established range for XLM-R-large is ~5e-6–2e-5; fine-tuning instability grows with model size (Mosbach et al., 2021), which also motivates the collapse guard |
| optimiser | AdamW | Loshchilov and Hutter (2019) |
| `max_len` | 128 | data-driven (median 8, mean 9 tokens) |

Searched per architecture, identical grid, exhaustively (24 configurations,
Optuna `GridSampler` — every point is visited, so no architecture can be luckier
than another and the sampler seed is irrelevant):

| parameter | grid | source |
|---|---|---|
| `head_lr` | {1e-4, 5e-4, 1e-3, 2e-3} | Howard and Ruder (2018) — a newly initialised head needs a higher rate than the pretrained body. Spans 10x–200x the encoder rate: the baseline's 1.05M-parameter MLP head is conventionally trained near the encoder rate, the 7.84M from-scratch CNN-BiLSTM head wants far more, and the grid has to contain both optima |
| `dropout` | {0.1, 0.3} | 0.1 is the BERT default. 0.5 is excluded because the dual-view head applies dropout at three points before the classifier, leaving an effective retention near 0.125. Note this knob is not the same intervention in both architectures — the baseline applies it once, in `classifier_dropout` |
| `epochs` | {3, 4, 5} | Devlin et al. (2019) {2,3,4} shifted up by one, since the encoder learns at a deliberately gentle rate; 741/988/1,235 optimizer steps at batch 32 |

## Repository layout

```
config.py                 central config (seeds, fractions, fixed HPs, search space)
utils_split.py            hashing / split helpers
requirements.txt
notebooks/                the full pipeline (see execution order below)
splits/                   split manifest and fold map (generated)
hpo/                      best_params.json per variant + Optuna study (generated)
cv/                       per-variant summaries, OOF predictions, figures (generated)
test_eval/                descriptive + significance CSVs (generated)
analysis/                 gate saturation, category error rates, figures (generated)
annotation/               re-annotation round builder + metadata
datasets/                 (corpus not included — see datasets/README.md)
```

## Data

The KurdiSent corpus is **not** redistributed here. Obtain it from the original
source and regenerate the derived data locally; the seeds make this
deterministic, and `splits/split_manifest.json` records SHA-256 hashes so you can
verify your regeneration. See [`datasets/README.md`](datasets/README.md).

## Setup

```bash
pip install -r requirements.txt
```

The notebooks are written for **Google Colab** (they mount Google Drive). To run
elsewhere, set the `KUSA_ROOT` environment variable to the checkout directory (or
adjust the `sys.path.insert(...)` line at the top of each notebook) so that
`from config import *` resolves; `config.py` is otherwise self-locating.

## Execution order

```
0. preprocess_kurdisent_v2        -> datasets/          (once, needs KLPT 0.1.7)
1. build_split_v2                 -> splits/            (once)
2. hpo_kusa_dual_view_v2,
   hpo_kusa_baseline_v2           -> hpo/  (exhaustive grid, 24 configurations each)
3. hpo_kusa_dual_view_dupinput_v2,
   hpo_kusa_dual_view_gated_v2    -> inherit from dual_view (no training)
4. kusa_*_cv_v2 (4x)             -> cv/ (5 fold models per variant) + OOF
5. kusa_category_error_outlier_v2, gate_analysis_v2,
   kappa_evaluation_both_rounds   -> analysis/
6. test_evaluation_v2            -> test_eval/   ONLY AT THE VERY END
7. aso_offline_v2                -> test_eval/   (from stored predictions)
```

`new_klpt_analysis_n_comparison` produces the KLPT coverage figures and can run
any time after preprocessing.

## Statistics

- **descriptive**: macro-F1 of the five fold models per architecture, mean ± std.
- **paired over items** (primary): bootstrap CI of the macro-F1 difference on the
  averaged soft-vote predictions, plus an exact McNemar test on those same
  predictions. Both are independent across the 2,455 test items.
- **paired over seeds** (secondary): paired t-test over the five fold
  differences. The folds share ~75% of their training data, so this test is
  anti-conservative on its own (Dietterich, 1998) and is reported as support, not
  as the primary evidence.

## Rules

- `splits/test_LOCKED.csv` is read exclusively by `test_evaluation_v2`; there the
  stored fold models score the test set (pure inference, no selection, fixed
  epoch count).
- The HPO slice uses `HPO_SLICE_SEED` (not the fold seed) so it does not coincide
  with fold 0, and is built from the same rows for every search.
- Every architecture trains its randomly initialised head in its own parameter
  group at its own searched rate.
- Seeds, fractions, the fixed encoder hyperparameters and the head search space
  live in `config.py`, not in the notebooks.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The KurdiSent corpus is a separate work by
Badawi et al. (2025) with its own terms and is not covered by this license.

## Citation

If you use this code, please cite our paper (details to follow) and the KurdiSent
corpus:

> Soran Badawi, Arefeh Kazemi, Vali Rezaie (2025). KurdiSent: a corpus for
> Kurdish sentiment analysis. Language Resources and Evaluation 59(1), 601-620.
