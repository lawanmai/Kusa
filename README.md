# kusa — Dual-View SARF for Sorani Kurdish Sentiment Analysis

Code and data for the paper:

> **Does Morphological Multi-View Transfer to Low-Resource Kurdish?
> A Dual-View Adaptation of SARF and a Re-Annotation Audit of KurdiSent**
> Lawan Mai

---

## Overview

This repository contains:

- **Preprocessing** of KurdiSent with KLPT morphological analysis
- **Hyperparameter search** (Optuna) for all three model variants
- **Cross-validation** for the single-view XLM-RoBERTa baseline and both dual-view variants (average fusion and gated fusion)
- **Error analysis** and category-level outlier detection
- **Re-annotation sampling** and inter-annotator agreement evaluation (Cohen's κ, PABAK, Wilson CIs)

---

## Data

### KurdiSent

The corpus used for all experiments is **KurdiSent** ([Badawi et al., 2025](https://doi.org/10.1007/s10579-023-09716-6)), a collection of 12,306 Sorani Kurdish texts annotated for sentiment (negative / neutral / positive). KurdiSent is **not included** in this repository. Please obtain it directly from the authors and place it as `KurdiSent.csv` in the working directory before running any notebook.

### Re-annotation data

The `data/` folder contains the re-annotation artifacts with **text columns removed** out of respect for the KurdiSent license:

- `annotation_import_labelstudio.csv` — the 745 item IDs that were imported into Label Studio
- `labelstudio_export.csv` — the blind linguist labels and metadata (annotation_id, sentiment, lead_time, note, timestamps), without the original texts
- `reannotation_summary.csv` — the aggregated agreement statistics (n, Po, disagreement rate, κ with bootstrap CIs, PABAK) for the representative and News arms

To reproduce the agreement evaluation, join these files with KurdiSent via the `id` column.

---

## Reproducing the Experiments

All notebooks were developed and run on **Google Colab** with a GPU runtime (L4). They read from and write to Google Drive; adjust the `DIR_PATH` variable at the top of each notebook to point to your own Drive folder.

## Requirements

The notebooks run on Google Colab and install dependencies inline. The main packages are:

- `transformers`
- `torch`
- `scikit-learn`
- `optuna`
- `klpt`
- `pandas`, `numpy`, `matplotlib`

---

## Citation

If you use this code or the re-annotation data, please cite:

```bibtex
@article{mai-2026-kurdish-dualview,
  title   = {Does Morphological Multi-View Transfer to Low-Resource {K}urdish?
             {A} Dual-View Adaptation of {SARF} and a Re-Annotation Audit of {K}urdi{S}ent},
  author  = {Mai, Lawan},
  year    = {2026},
  note    = {Under review}
}
```


---

## License

The code in this repository is released under the **MIT License**. KurdiSent itself is not part of this repository; its use is subject to the terms of Badawi et al. (2025).