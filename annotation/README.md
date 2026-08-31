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
| audit | 99 | confident errors of the baseline as trained under the initial protocol | no |

The three arms overlap by exactly four items: the audit arm holds 99 items, four
of which were also drawn into the representative arm and are counted there, so
95 items are audit-only. 500 + 150 + 95 = 745.

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
| `round1_labels_public.csv` | per-item labels for all 745 round-1 items, **without the sentences** |
| `make_public_labels.py` | rebuilds that file from the private round-1 artefacts |

`round1_labels_public.csv` is the audit released as a resource. Per item it gives
the identifier, category, arm membership, the published gold label and the two
annotators' independent labels — nine columns drawn from a closed vocabulary, no
sentence and no annotator free-text note. That makes it distributable without
redistributing KurdiSent: obtain the corpus, join on `id`, and every agreement
figure in the paper follows. `make_public_labels.py` asserts on the way out that
no value outside `{Negative, Neutral, Positive}` reaches the file.

## What is withheld, and why

`round1/` and `round2/` are excluded in full by `.gitignore`. Every file in them —
the Label Studio imports and exports, the merged label tables, the gold keys —
embeds the KurdiSent sentences the annotators judged. The corpus is a separate
work by Badawi et al. (2025) under its own terms and is not redistributed here,
in whole or in part, so the annotation rounds cannot be published either. The
`*_key_PRIVATE.csv` gold keys are additionally withheld because they would let
anyone reconstruct which items the blind arms contained.

`round1_labels_public.csv` is the one exception, and it is an exception precisely
because it carries no text: it is derived from those withheld files but keeps
only identifiers and labels.

This is why `notebooks/kappa_evaluation_both_rounds.ipynb` cannot be re-run from
a fresh checkout: the notebook and its aggregate output are released, but its
inputs are not. The agreement figures themselves can now be recomputed from
`round1_labels_public.csv` alone, without the corpus. To rebuild the rounds,
obtain the corpus, then run

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
