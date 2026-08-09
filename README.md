# kusa — Morphology-Aware Sentiment Analysis for Sorani Kurdish

Code and reproducibility artefacts for a dual-view adaptation of the SARF
architecture to Sorani Kurdish sentiment analysis on the KurdiSent corpus,
evaluated under a leakage-controlled held-out protocol.

A statistical design with seed pairing and minimal HPO. When the expected effect
is below one F1 point, the adversary is seed noise, not the amount of data — so
the compute budget goes into repetitions, not into a hyperparameter search.

- 20% test set, drawn group-aware and locked away by SHA-256.
- Grouped 5-fold CV on the dev pool, with an identical fold assignment for all
  architectures.
- Encoder hyperparameters fixed; only the head is searched (on `dual_view_v2`)
  and transferred to baseline and gated → an identical training setup isolates
  the fusion.
- Fixed epoch count, no early stopping, collapse guard (loss > ln 3 → restart).
- The **five fold models** of each architecture score the test set → a spread
  estimate without extra runs.

## Results (held-out test set, n = 2,455, macro-F1)

| Model | Macro-F1 (ensemble) | Per-fold (mean ± std) | Accuracy |
|---|---|---|---|
| XLM-R-large, single-view (baseline) | 0.811 | 0.795 ± 0.009 | 0.808 |
| Dual-view SARF, average fusion | 0.839 | 0.825 ± 0.009 | 0.838 |
| Dual-view SARF, gated fusion *(ablation)* | 0.842 | 0.827 ± 0.005 | 0.840 |

Both dual-view variants beat the baseline by ~3 macro-F1 points, significant on
every test (paired t p < 0.01, item-level bootstrap CI excluding 0, exact
McNemar p < 0.001, almost-stochastic-order dominance). The two fusion mechanisms
are statistically indistinguishable; the gate saturates to a single view per
fold, so the gated variant is used only as a probe of view redundancy.

## Repository layout

```
config.py                 central config (seeds, fractions, fixed encoder HPs)
utils_split.py            hashing / split helpers
requirements.txt
notebooks/                the full pipeline (see execution order below)
splits/                   split_manifest.json (SHA-256), distribution, dev_folds
hpo/                      best_params.json per variant + Optuna study
cv/                       per-variant cv_summary, error rates, OOF preds, confusion
test_eval/                descriptive + significance CSVs, per-fold test preds
analysis/                 gate saturation, category error rates, robustness, figures
annotation/              re-annotation round builder + metadata
datasets/                 (data not included — see datasets/README.md)
```

## Data

The KurdiSent corpus is **not** redistributed here. Obtain it from the original
source and regenerate the derived data locally; the seeds make this
deterministic and the committed `splits/split_manifest.json` lets you verify the
result by SHA-256. See [`datasets/README.md`](datasets/README.md).

## Setup

```bash
pip install -r requirements.txt
```

The notebooks are written for **Google Colab** (they mount Google Drive). To run
elsewhere, set the `KUSA_ROOT` environment variable to the checkout directory (or
adjust the `sys.path.insert(...)` line at the top of each notebook) so that
`from config import *` resolves; `config.py` is otherwise self-locating.

## Execution order

1. `build_split_v2`                 → splits/            (once)
2. `hpo_kusa_dual_view_v2`          → hpo/dual_view_v2/  (the only search, ~10 trials)
3. `hpo_kusa_baseline_v2`,
   `hpo_kusa_dual_view_gated_v2`    → derive their parameters from dual_view (no training)
4. `kusa_*_cv_v2` (3x)             → cv/ (5 fold models per variant) + OOF
5. `kusa_category_error_outlier_v2`, `kappa_evaluation_both_rounds` → analysis/
6. `robustness_baseline_epochs_v2` → analysis/ (optional, dev pool only)
7. `test_evaluation_v2`            → test_eval/   ONLY AT THE VERY END

`preprocess_kurdisent_v2` and `new_klpt_analysis_n_comparison` handle
preprocessing and the KLPT coverage analysis; `aso_offline_v2` recomputes the
ASO test from stored predictions without any extra dependency.

## Statistics (test_evaluation_v2)

- **descriptive**: macro-F1 of the five fold models per architecture, mean ± std.
- **paired over seeds**: paired t-test + ASO (deep-significance) over the five
  fold differences.
- **paired over items**: bootstrap CI of the macro-F1 difference on the averaged
  (soft-vote) predictions + exact McNemar on the majority votes.

## Rules

- `splits/test_LOCKED.csv` is read exclusively by `test_evaluation_v2`; there the
  stored fold models score the test set (pure inference, no selection, fixed
  epoch count).
- The HPO slice uses `HPO_SLICE_SEED` (not the fold seed) so that it does not
  coincide with fold 0.
- Seeds, fractions, and the fixed encoder hyperparameters live in `config.py`,
  not in the notebooks.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The KurdiSent corpus is a separate work by
Badawi et al. (2025) with its own terms and is not covered by this license.

## Citation

If you use this code, please cite our paper (details to follow) and the KurdiSent
corpus:

> Soran Badawi, Arefeh Kazemi, Vali Rezaie (2025). KurdiSent: a corpus for
> Kurdish sentiment analysis. Language Resources and Evaluation 59(1), 601-620.
