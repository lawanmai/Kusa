# datasets/

The **KurdiSent corpus is not redistributed in this repository.** It is a
separate work by Badawi et al. (2025) and is covered by its own license/terms.
This folder ships only the metadata needed to reproduce our pipeline; the
text-bearing files are git-ignored and must be regenerated locally.

## How to obtain the corpus

Get `KurdiSent.csv` from the original source:

> Soran Badawi, Arefeh Kazemi, Vali Rezaie (2025).
> *KurdiSent: a corpus for Kurdish sentiment analysis.*
> Language Resources and Evaluation 59(1), 601-620.
> https://doi.org/10.1007/s10579-023-09716-6

Place the file at `datasets/KurdiSent.csv` (columns: `num_label, category, text`).

## How to regenerate the derived data

Run, in order:

1. `notebooks/preprocess_kurdisent_v2.ipynb`
   -> writes `datasets/KurdiSent_preprocessed.csv` (adds `surface`, `lemma` via KLPT)
2. `notebooks/build_split_v2.ipynb`
   -> writes `splits/dev_pool.csv`, `splits/test_LOCKED.csv`, `splits/dev_folds.csv`

Both steps are deterministic (seeds in `config.py`). Verify your regeneration
against the committed `splits/split_manifest.json`, which records the SHA-256
hashes of the source, dev pool, test set, and fold files. If the hashes match,
your local data is bit-identical to what produced the paper's results.

## What IS committed here

- `preprocessing_meta.json` — KLPT version, row counts, and the SHA-256 of the
  corpus going in and of the preprocessed file coming out
- `../splits/split_manifest.json` — SHA-256 hashes + split metadata
- `../splits/split_distribution.csv` — label x category counts per split
- `../splits/dev_folds.csv` — `row_id -> fold` map (integers only, no text)

Everything else in `datasets/` and the text-bearing split files stay local. This
is enforced by `.gitignore`, not by discipline: `datasets/*.csv` and
`splits/*.csv` are denied wholesale and the two safe split files are re-included
by name. A new file dropped into either folder is therefore unpublished until
someone adds an explicit exception — check the corpus terms before you do.

## Verifying a regeneration

```bash
sha256sum datasets/KurdiSent.csv datasets/KurdiSent_preprocessed.csv           splits/dev_pool.csv splits/test_LOCKED.csv splits/dev_folds.csv
```

Compare against `source_sha256` / `output_sha256` in `preprocessing_meta.json`
and against `source_sha256`, `dev_pool_sha256`, `test_locked_sha256` and
`dev_folds_sha256` in `../splits/split_manifest.json`. All five matching means
your local data is bit-identical to what produced the published results.
