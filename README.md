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

> **Status:** the pipeline has run end to end and the result files are committed.
> The locked test set has been scored once and is now spent.

## Results

Test set, n = 2,455, five-fold soft-vote ensemble (`test_eval/test_results.csv`):

| variant | test macro-F1 | per-fold mean ± std | dev-pool OOF |
|---|---|---|---|
| `baseline_v2` | 0.814 | 0.806 ± 0.004 | 0.806 |
| `dual_view_dupinput_v2` (control) | 0.834 | 0.822 ± 0.007 | 0.809 |
| `dual_view_v2` | **0.838** | 0.828 ± 0.005 | 0.814 |
| `dual_view_gated_v2` | 0.832 | 0.824 ± 0.006 | 0.813 |

The decomposition the control makes possible (`test_eval/significance_*.csv`):

| step | Δ macro-F1 | bootstrap 95% CI | McNemar | paired t over folds |
|---|---|---|---|---|
| architecture (`baseline_v2` → control) | +0.020 | [+0.011, +0.030] | p < 0.001 | p = 0.025 |
| morphology (control → `dual_view_v2`) | +0.004 | [−0.003, +0.010] | p = 0.41 | p = 0.095 |
| total (`baseline_v2` → `dual_view_v2`) | +0.024 | [+0.014, +0.033] | p < 0.001 | p = 0.002 |

The dual-view architecture improves over the single-view baseline by about two
macro-F1 points, and that gain survives every paired test. Replacing the second
branch's surface view with the lemma view adds a further +0.004, which does not.
With capacity and configuration held fixed, the measurable gain is architectural;
the morphological contribution specifically is small and not separable from noise
on this corpus.

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
models.py                 datasets and architectures, defined once for every notebook
utils_split.py            hashing / split helpers
requirements.txt
notebooks/                the full pipeline (see execution order below)
splits/                   split manifest, fold map, label distribution (generated)
hpo/                      best_params.json per variant + Optuna study (generated)
cv/                       per-variant summaries, OOF predictions, figures (generated)
test_eval/                descriptive + significance CSVs (generated)
analysis/                 gate saturation, category error rates, figures (generated)
annotation/               re-annotation builder, guidelines PDF, agreement summary,
                          text-free per-item labels
datasets/                 preprocessing metadata only (corpus not included)
```

## Data

The KurdiSent corpus is **not** redistributed here. It is a separate work by
Badawi et al. (2025) under its own terms, and nothing in this repository
reproduces its sentences. Obtain the corpus from the original source and
regenerate the derived data locally; the seeds make this deterministic, and
`splits/split_manifest.json` records SHA-256 hashes so you can verify your
regeneration bit-for-bit. See [`datasets/README.md`](datasets/README.md).

**Published** (text-free throughout — identifiers, labels, predictions,
probabilities and aggregate statistics only):

| path | contents |
|---|---|
| `datasets/preprocessing_meta.json` | KLPT version, row counts, SHA-256 of corpus in and out |
| `splits/split_manifest.json` | split seed, sizes, SHA-256 of all four generated files |
| `splits/dev_folds.csv` | `row_id -> fold` map for the development pool |
| `splits/split_distribution.csv` | label x category counts per split |
| `hpo/*/best_params.json` | the winning configuration per variant (the two control arms inherit dual-view's) |
| `hpo/baseline_v2/study.db`, `hpo/dual_view_v2/study.db` | the two Optuna studies in full, all 24 trials each |
| `cv/*/` | fold summaries, OOF predictions with class probabilities, confusion matrices |
| `test_eval/` | the one-time test results and every significance test |
| `analysis/` | category error rates, gate saturation, KLPT coverage |
| `annotation/annotation_guidelines.pdf` | the guidelines both annotators worked from |
| `annotation/agreement_summary_both_rounds.csv` | kappa, PABAK and CIs per comparison |
| `annotation/round1_labels_public.csv` | per-item gold and both annotators' labels for all 745 round-1 items, without the sentences |

**Not published**, and excluded by `.gitignore` rather than by hand: the corpus
and everything derived from it that carries text (`datasets/*`,
`splits/dev_pool.csv`, `splits/test_LOCKED.csv`, the whole
`annotation/round1/` and `annotation/round2/` directories, `*_PRIVATE.csv`
gold keys, `*_misclassified.csv`, `oof_surface_weights.csv`), plus the 20 fold
checkpoints. The ignore rule denies these paths by default and re-includes the
safe files by name, so a new file under `datasets/`, `splits/` or `annotation/`
is unpublished until someone adds an explicit exception.

One consequence for reproducibility: `kappa_evaluation_both_rounds` reads the
annotators' label files, which carry the sentences they judged and therefore stay
local. Re-running that notebook end to end requires rebuilding the rounds with
`annotation/build_annotation_rounds.py` from your own copy of the corpus. The
agreement figures themselves, however, can be recomputed without the corpus from
`annotation/round1_labels_public.csv`, which carries the per-item labels but none
of the text. Every other number in the paper comes from files committed here.

## Setup

```bash
pip install -r requirements.txt
```

The notebooks are written for **Google Colab** (they mount Google Drive). To run
elsewhere, set the `KUSA_ROOT` environment variable to the checkout directory (or
adjust the `sys.path.insert(...)` line at the top of each notebook) so that
`from config import *` resolves; `config.py` is otherwise self-locating.
`annotation/build_annotation_rounds.py` follows the same convention and resolves
the root from its own location when `KUSA_ROOT` is unset. No path in this
repository is tied to a particular machine or Drive account.

## Execution order

```
0. preprocess_kurdisent_v2        -> datasets/          (once, needs KLPT 0.1.7)
1. build_split_v2                 -> splits/            (once)
2. hpo_kusa_dual_view_v2,
   hpo_kusa_baseline_v2           -> hpo/  (exhaustive grid, 24 configurations each)
3. hpo_kusa_dual_view_dupinput_v2,
   hpo_kusa_dual_view_gated_v2    -> inherit from dual_view (no training)
4. kusa_*_cv_v2 (4x)             -> cv/ (5 fold models per variant) + OOF
5. kusa_category_error_outlier_v2, gate_analysis_v2  -> analysis/
   kappa_evaluation_both_rounds                      -> annotation/
6. test_evaluation_v2            -> test_eval/   ONLY AT THE VERY END
```

`new_klpt_analysis_n_comparison` produces the KLPT coverage figures and can run
any time after preprocessing.

## Statistics

- **descriptive**: macro-F1 of the five fold models per architecture, mean ± std.
- **paired over items** (primary): bootstrap CI of the macro-F1 difference on the
  averaged soft-vote predictions, plus an exact McNemar test on those same
  predictions. Both are independent across the 2,455 test items.
- **paired over seeds** (secondary): paired t-test over the five fold
  differences, plus the Almost-Stochastic-Order test (`deepsig`). The folds share
  ~75% of their training data, so the t-test is anti-conservative on its own
  (Dietterich, 1998), and with n = 5 runs ASO cannot carry a dominance claim
  either. Both are reported as support, not as the primary evidence.

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

Code and the generated result files: MIT (see [`LICENSE`](LICENSE)).

The KurdiSent corpus is a separate work by Badawi et al. (2025) with its own
terms and is **not** covered by this license and **not** contained in this
repository, in whole or in part. If you obtain the corpus to reproduce these
results, its terms govern your use of it.

## Citation

If you use this code or the released result files, please cite both the paper
this repository accompanies and the corpus it is built on. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff).

> Lawan Mai, Shene Hassan. *Morphology-Aware Sentiment Analysis for Sorani
> Kurdish: A Dual-View Adaptation of SARF.* Under review; venue and year to be
> filled in on acceptance.

> Soran Badawi, Arefeh Kazemi, Vali Rezaie (2025). KurdiSent: a corpus for
> Kurdish sentiment analysis. Language Resources and Evaluation 59(1), 601-620.
> https://doi.org/10.1007/s10579-023-09716-6
