"""Common helpers for text normalization and group-aware splits."""
import hashlib
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def norm(s):
    """Remove NFKC, ZWNJ, and whitespace, standardize Arabic variants."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u200c", "")
    s = re.sub(r"\s+", "", s)
    return s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")


def add_group_and_stratum(df, text_col="text", label_col="label",
                          cat_col="category"):
    """Adds 'group' (normalized text) and 'stratum' (label x category)."""
    out = df.copy()
    out["group"] = out[text_col].map(norm)
    out["stratum"] = out[label_col].astype(str) + "_" + out[cat_col].astype(str)
    return out


def grouped_holdout(df, frac, seed, group_col="group", stratum_col="stratum"):
    """A group-aware, stratified holdout split.

    frac is rounded to the nearest 1/n_splits, since
    StratifiedGroupKFold only produces fractions of this form.
    """
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, df[stratum_col], groups=df[group_col]))
    return df.iloc[keep_idx].copy(), df.iloc[hold_idx].copy()


def assign_folds(df, n_folds, seed, group_col="group", stratum_col="stratum"):
    """Returns a Series with the fold number (0..n_folds-1) per row."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed)
    folds = pd.Series(-1, index=df.index, dtype=int)
    for k, (_, va) in enumerate(sgkf.split(df, df[stratum_col],
                                           groups=df[group_col])):
        folds.iloc[va] = k
    assert (folds >= 0).all(), "Not all rows assigned to a fold"
    return folds


def no_group_overlap(a, b, group_col="group"):
    """True if no group appears in both subsets."""
    return not (set(a[group_col]) & set(b[group_col]))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def distribution(df, label_col="label", cat_col="category"):
    """label x category crosstab, for the appendix."""
    return pd.crosstab(df[label_col], df[cat_col], margins=True)

