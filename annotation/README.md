# annotation/

A blind re-annotation audit of the KurdiSent labels, run alongside the modelling
experiments. Two annotators independently relabelled a sample of the corpus; the
audit asks how far the corpus gold labels are reproducible, and how much of the
models' remaining error is label noise rather than model error.

## Design

**Round 1** — 745 items, both annotators, the same batch in a different order.

| arm | n | drawn from | evaluated |
|---|---|---|---|
| representative | 500 | full corpus, proportional to label x category | yes |
| news top-up | 150 | news only, disjoint from arm 1 | yes |
| audit | 95 | confident errors of the baseline as trained under the initial protocol | no |

Only arms 1 and 2 (650 items) enter the agreement statistics. The audit arm is
batch context and is reported as counts, never as a rate: it was drawn from a
pre-split full-corpus run, so its sampling frame no longer matches the model that
produced the published results.

**Round 2** — 250 items sampled proportionally from the locked test set, disjoint
from every text shown in round 1, intended as a human ceiling on the test set.
**Round 2 is parked**: it was built and handed out but has not been returned, so
no round-2 rows appear in the agreement summary and no human ceiling is claimed.

## What is published here

| file | contents |
|---|---|
| `annotation_guidelines.pdf` | the guidelines both annotators worked from |
| `build_annotation_rounds.py` | rebuilds every round artefact from primary sources |
| `annotation_build_meta.json` | seeds, arm sizes, SHA-256 of the inputs each round was drawn from |
| `agreement_summary_both_rounds.csv` | per-comparison `n`, observed agreement, kappa with CI, PABAK |

## What is withheld, and why

`round1/` and `round2/` are excluded in full by `.gitignore`. Every file in them —
the Label Studio imports and exports, the merged label tables, the gold keys —
embeds the KurdiSent sentences the annotators judged. The corpus is a separate
work by Badawi et al. (2025) under its own terms and is not redistributed here,
in whole or in part, so the annotation rounds cannot be published either. The
`*_key_PRIVATE.csv` gold keys are additionally withheld because they would let
anyone reconstruct which items the blind arms contained.

This is why `notebooks/kappa_evaluation_both_rounds.ipynb` cannot be re-run from
a fresh checkout: the notebook and its aggregate output are released, but its
inputs are not. To reproduce it, obtain the corpus, then run

```bash
python annotation/build_annotation_rounds.py
```

which regenerates the exact batches from the recorded seeds and hashes. Nothing
in that script reads an annotation result — the linguists' labels are never an
input to the sampling, so a rebuild cannot be contaminated by them. Round 1 is
rebuilt and *verified* against existing files rather than overwritten; set
`WRITE_ROUND1 = True` only when bootstrapping a fresh project.

The script resolves the project root from its own location, or from `KUSA_ROOT`
if set.

## Reading the summary

`agreement_summary_both_rounds.csv` reports each comparison twice over: `kappa`
with a bootstrap CI, and `PABAK` (prevalence-adjusted, bias-adjusted kappa)
alongside raw agreement `Po`. The two diverge where the label distribution is
skewed, which is exactly the situation in the news slice — read them together,
not in isolation.
