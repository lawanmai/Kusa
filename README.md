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
  head runs its **own** search over the same space with the same budget. Equal
  search budgets, not a shared configuration, are what make the comparison fair.
- Epoch count searched per architecture, no early stopping, collapse guard
  (loss > ln 3 → restart with a different seed).
- The **five fold models** of each architecture score the test set → a spread
  estimate without extra runs, plus paired significance tests over seeds and
  over items.

> **Status:** results are being regenerated after a protocol correction (the
> baseline's classification head now trains in its own parameter group at its own
> searched learning rate, and a single-view ablation was added). This repository
> currently ships the code and the protocol; result files will be committed once
> the runs complete.

## Architectures

| variant | head | HPO |
|---|---|---|
| `baseline_v2` | XLM-R classification head | own search |
| `single_view_complex_v2` | CNN-BiLSTM, surface only (ablation) | own search |
| `dual_view_v2` | cross-attention + CNN-BiLSTM | own search |
| `dual_view_gated_v2` | same + gate (probe) | inherits `dual_view_v2` |

`single_view_complex_v2` separates the effect of the CNN-BiLSTM head from the
effect of the second view: without it, a gain over the baseline is confounded
between the two. It keeps the encoder, CNN (kernels 3/4/5, 200 filters), BiLSTM
(128 per direction), dropout placement and classifier of the dual-view model, and
removes only the lemma view and the cross-attention. It runs its own search,
pre-registered rather than conditional: this is the run that has to be able to
refute the claim that the second view matters, so it competes in its best
configuration rather than in one tuned for a two-view input.

## Hyperparameters

Fixed and identical for every architecture:

| parameter | value | source |
|---|---|---|
| `batch_size` | 32 | Devlin et al. (2019), App. A.3 |
| `weight_decay` | 0.01 | Liu et al. (2019) |
| `warmup_ratio` | 0.10 | Liu et al. (2019) |
| `encoder_lr` | 1e-5 | Mosbach et al. (2021) — below the BERT grid, since large models destabilise at higher rates |
| optimiser | AdamW | Loshchilov and Hutter (2019) |
| `max_len` | 128 | data-driven (median 8, mean 9 tokens) |

Searched per architecture, same space and same budget (10 TPE trials, Optuna):

| parameter | grid | source |
|---|---|---|
| `head_lr` | {5e-4, 1e-3, 2e-3} | Howard and Ruder (2018) — a newly initialised head needs a higher rate than the pretrained body |
| `dropout` | {0.1, 0.3, 0.5} | standard regularisation range |
| `epochs` | {3, 4, 5} | brackets Devlin et al. (2019) |

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
0. preprocess_kurdisent_v2                        -> datasets/   (needs KLPT 0.1.7)
1. build_split_v2                                 -> splits/
2. hpo_kusa_dual_view_v2, hpo_kusa_baseline_v2    -> hpo/        (10 trials each)
3. hpo_kusa_single_view_complex_v2,
   hpo_kusa_dual_view_gated_v2                    -> hpo/        (inherit, no training)
4. kusa_*_cv_v2 (4x)                              -> cv/         (5 fold models each)
5. kusa_category_error_outlier_v2, gate_analysis_v2,
   kappa_evaluation_both_rounds                   -> analysis/
6. test_evaluation_v2                             -> test_eval/  ONLY AT THE VERY END
7. aso_offline_v2                                 -> test_eval/
```

`new_klpt_analysis_n_comparison` produces the KLPT coverage figures and can run
any time after preprocessing.

## Statistics

- **descriptive**: macro-F1 of the five fold models per architecture, mean ± std.
- **paired over items** (primary): bootstrap CI of the macro-F1 difference on the
  averaged soft-vote predictions, plus an exact McNemar test on the majority
  votes. These are independent across the 2,455 test items.
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
