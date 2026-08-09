"""
KurdiSent re-annotation — reproducible builder for both rounds.

Regenerates every annotation artefact from primary sources. Nothing is derived
from an annotation result; the linguists' labels are never read.

  Round 1  (745 items, two annotators, same batch in different order)
      arm 1  representative : 500 items, proportional to label x category
      arm 2  news top-up    : 150 extra news items, disjoint from arm 1
      arm 3  audit          :  95 confident baseline errors (conf >= 0.99)
      Only arms 1 and 2 (650 items) are evaluated. Arm 3 is batch context: it
      was drawn from a pre-split full-corpus run and carries no claim.

  Round 2  (250 items, two annotators, same batch in different order)
      test noise ceiling    : proportional random sample of the locked test set,
                              disjoint from every text shown in round 1.

Round 1 is REBUILT AND VERIFIED against the files already in the project, never
silently overwritten — annotator A has worked through that exact batch, and a
different shuffle would invalidate it. Set WRITE_ROUND1 = True only when
bootstrapping a fresh project.

Usage in Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !python build_annotation_rounds.py
"""
import hashlib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
ROOT = "/content/drive/MyDrive/google_colab/kusa/v2_heldout"

DATA       = ROOT + "/datasets"
SPLITS     = ROOT + "/splits"
ANNOTATION = ROOT + "/annotation"
R1_DIR     = ANNOTATION + "/round1"
R2_DIR     = ANNOTATION + "/round2"

ORIG        = DATA + "/KurdiSent.csv"
PRE         = DATA + "/KurdiSent_preprocessed.csv"
MIS         = DATA + "/kusa_baseline_cv_inc_error_analysis_alternative_misclassified.csv"
TEST_LOCKED = SPLITS + "/test_LOCKED.csv"
DEV_POOL    = SPLITS + "/dev_pool.csv"
MANIFEST    = SPLITS + "/split_manifest.json"

# round 1 — must not change, these reproduce the batch annotator A worked through
SEED_R1        = 42
N_REPRESENT    = 500
N_NEWS_TOPUP   = 150
AUDIT_CONF_MIN = 0.99

# round 2
SEED_R2   = 2026
N_CEILING = 250

WRITE_ROUND1 = False   # True only when no round-1 files exist yet

NAME = {0: "Neutral", 1: "Negative", 2: "Positive"}
INV  = {v: k for k, v in NAME.items()}


# ----------------------------------------------------------------- helpers
def norm(s):
    """Matching key: NFKC, drop ZWNJ and whitespace, unify Arabic variants."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u200c", "")
    s = re.sub(r"\s+", "", s)
    return s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def largest_remainder(counts, total):
    """Proportional allocation, leftovers to the largest fractional parts."""
    exact = counts / counts.sum() * total
    alloc = np.floor(exact).astype(int)
    rest  = total - int(alloc.sum())
    if rest > 0:
        alloc[(exact - alloc).sort_values(ascending=False).index[:rest]] += 1
    return alloc


def allocate_exact(counts, total):
    """Like largest_remainder, but caps at capacity and redistributes the
    shortfall so the draw lands on exactly `total` items."""
    alloc = largest_remainder(counts, total).clip(upper=counts)
    for _ in range(50):
        short = total - int(alloc.sum())
        if short <= 0:
            break
        free = (counts - alloc)
        free = free[free > 0]
        if free.empty:
            break
        for stratum in (free / free.sum()).sort_values(ascending=False).index:
            if short == 0:
                break
            take = min(short, int(free[stratum]))
            alloc[stratum] += take
            short -= take
    return alloc


def write_pair(df, out_dir, tag, seed_a, seed_b):
    """Write the two annotator imports: same ids, different row order."""
    os.makedirs(out_dir, exist_ok=True)
    a = df[["id", "text"]].sample(frac=1.0, random_state=seed_a)
    b = df[["id", "text"]].sample(frac=1.0, random_state=seed_b)
    pa = f"{out_dir}/{tag}_import_annotatorA.csv"
    pb = f"{out_dir}/{tag}_import_annotatorB.csv"
    a.to_csv(pa, index=False, encoding="utf-8-sig")
    b.to_csv(pb, index=False, encoding="utf-8-sig")
    same = int((a["id"].values == b["id"].values).sum())
    print(f"    {pa}  ({len(a)} items)")
    print(f"    {pb}  ({len(b)} items)")
    print(f"    identical positions between A and B: {same}")
    return pa, pb


# ================================================================= round 1
def build_round1():
    print("=" * 70)
    print("ROUND 1 — 745-item batch (650 evaluated + 95 context)")
    print("=" * 70)

    for p in (ORIG, PRE, MIS):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")

    orig = pd.read_csv(ORIG, encoding="utf-8-sig")[["num_label", "category", "text"]].copy()
    orig["text"] = orig["text"].astype(str).str.strip()

    # texts carrying contradictory gold labels have no defined gold
    multi    = orig.groupby("text")["num_label"].nunique()
    conflict = set(multi[multi > 1].index)

    frame = (orig[~orig["text"].isin(conflict)]
             .drop_duplicates("text")
             .reset_index(drop=True))
    frame["key"]  = frame["text"].map(norm)
    frame_keys    = set(frame["key"])
    conflict_keys = {norm(t) for t in conflict}
    print(f"  frame: {len(frame)} unique texts "
          f"(excluded {len(conflict)} contradictory-label texts)")

    # surface -> raw, via the row-aligned preprocessed file
    pre = pd.read_csv(PRE, encoding="utf-8-sig")
    pre["sk"] = pre["surface"].map(norm)
    surf2raw  = pre.drop_duplicates("sk").set_index("sk")["text"].str.strip()

    # --- arm 1: representative, proportional to label x category
    cells = frame.groupby(["num_label", "category"]).size().rename("n").reset_index()
    cells["exact"] = cells["n"] / cells["n"].sum() * N_REPRESENT
    cells["base"]  = np.floor(cells["exact"]).astype(int)
    remainder = N_REPRESENT - int(cells["base"].sum())
    cells = cells.sort_values("exact", ascending=False).reset_index(drop=True)
    frac_order = (cells["exact"] - np.floor(cells["exact"])).sort_values(ascending=False).index
    for i in frac_order[:remainder]:
        cells.loc[i, "base"] += 1
    alloc = cells.set_index(["num_label", "category"])["base"].to_dict()

    rep_idx = []
    for (lab, cat), k in alloc.items():
        if k <= 0:
            continue
        pool = frame[(frame.num_label == lab) & (frame.category == cat)]
        rep_idx += pool.sample(n=min(k, len(pool)), random_state=SEED_R1).index.tolist()
    rep = frame.loc[rep_idx].copy()
    print(f"  arm 1 representative: {len(rep)} "
          f"(news within it: {(rep.category == 'news').sum()})")

    # --- arm 2: news top-up, disjoint from arm 1
    news_pool = frame[(frame.category == "news") & (~frame.index.isin(rep.index))]
    topup = news_pool.sample(n=min(N_NEWS_TOPUP, len(news_pool)),
                             random_state=SEED_R1 + 1).copy()
    print(f"  arm 2 news top-up   : {len(topup)} "
          f"-> news total {(rep.category == 'news').sum() + len(topup)}")

    # --- arm 3: audit, mapped back to raw text
    mis = pd.read_csv(MIS, encoding="utf-8-sig")
    audit = mis[mis["confidence"] >= AUDIT_CONF_MIN].copy()
    audit["sk"]  = audit["surface"].map(norm)
    audit["raw"] = audit["sk"].map(surf2raw)
    n_unmapped = int(audit["raw"].isna().sum())
    if n_unmapped:
        audit["raw"] = audit["raw"].fillna(audit["surface"].str.strip())
    audit["gold_num"] = audit["true_name"].map(INV)
    print(f"  arm 3 audit         : {len(audit)} at conf >= {AUDIT_CONF_MIN} "
          f"({n_unmapped} fell back to surface form)")

    # --- merge by raw text so nothing appears twice
    items = {}

    def add(text, gold, cat, arm, model_pred=None, model_conf=None, textform="raw"):
        k = norm(text)
        if k in items:
            items[k]["arm"].add(arm)
            if model_pred is not None:
                items[k]["model_pred"] = model_pred
                items[k]["model_conf"] = model_conf
        else:
            items[k] = dict(text=str(text).strip(), gold=int(gold), category=cat,
                            arm={arm}, model_pred=model_pred, model_conf=model_conf,
                            textform=textform)

    for _, r in rep.iterrows():
        add(r.text, r.num_label, r.category, "representative")
    for _, r in topup.iterrows():
        add(r.text, r.num_label, r.category, "news_topup")

    n_dropped = 0
    for _, r in audit.iterrows():
        fk = norm(r["raw"])
        if fk in conflict_keys:          # contradictory gold, excluded everywhere
            n_dropped += 1
            continue
        if fk in frame_keys:             # authoritative gold from the frame
            fr = frame[frame.key == fk].iloc[0]
            add(fr.text, fr.num_label, fr.category, "audit",
                r.pred_name, float(r.confidence))
        else:
            add(r["raw"], r.gold_num, r.category, "audit",
                r.pred_name, float(r.confidence),
                textform=("surface" if pd.isna(surf2raw.get(r["sk"], np.nan)) else "raw"))
    print(f"  audit dropped (contradictory gold): {n_dropped}")

    rows = [dict(text=v["text"], gold_label=v["gold"], gold_name=NAME[v["gold"]],
                 category=v["category"], arm=";".join(sorted(v["arm"])),
                 is_audit=int("audit" in v["arm"]), textform=v["textform"],
                 model_pred=v["model_pred"], model_conf=v["model_conf"])
            for v in items.values()]

    df = (pd.DataFrame(rows)
          .sample(frac=1.0, random_state=SEED_R1 + 2)
          .reset_index(drop=True))
    df.insert(0, "id", [f"item_{i:04d}" for i in range(len(df))])

    # evaluation columns: the four audit;representative items count as
    # representative, the audit-only items are context and never evaluated
    df["arm_eval"]     = df["arm"].replace({"audit;representative": "representative"})
    df["evaluated"]    = df["arm_eval"].isin(["representative", "news_topup"])
    df["context_only"] = ~df["evaluated"]

    print(f"  total {len(df)} items | evaluated {int(df['evaluated'].sum())} "
          f"| context {int(df['context_only'].sum())}")
    print(f"  arm_eval: {df.loc[df['evaluated'], 'arm_eval'].value_counts().to_dict()}")
    assert int(df["evaluated"].sum()) == 650, "expected 650 evaluated items"
    assert df["text"].map(norm).is_unique, "duplicate text in round 1"
    return df


def verify_round1(df):
    """Compare the rebuild with the key already in the project."""
    existing = f"{R1_DIR}/round1_key_PRIVATE.csv"
    if not os.path.exists(existing):
        print("  no existing round-1 key found - nothing to verify against")
        return None

    old = pd.read_csv(existing, encoding="utf-8-sig")
    a = df.set_index("id")["text"].map(norm)
    b = old.set_index("id")["text"].map(norm)
    same_ids  = set(a.index) == set(b.index)
    aligned   = a.reindex(b.index)
    identical = bool(same_ids and (aligned == b).all())

    print(f"  verification against {os.path.basename(existing)}")
    print(f"    same id set            : {same_ids}")
    if same_ids:
        n_diff = int((aligned != b).sum())
        print(f"    id -> text mismatches  : {n_diff}")
    print(f"    round 1 reproduces     : {identical}")
    if not identical:
        print("    WARNING: the rebuild differs from the batch already annotated.")
        print("    The existing files stay authoritative. Check that KurdiSent.csv,")
        print("    KurdiSent_preprocessed.csv and the misclassified export are the")
        print("    same versions the original sampler used.")
    return identical


def write_round1(df):
    os.makedirs(R1_DIR, exist_ok=True)
    write_pair(df, R1_DIR, "round1", SEED_R1 + 3, SEED_R2)
    cols = ["id", "gold_name", "gold_label", "category", "arm", "arm_eval",
            "evaluated", "context_only", "is_audit", "textform",
            "model_pred", "model_conf", "text"]
    p = f"{R1_DIR}/round1_key_PRIVATE.csv"
    df[cols].to_csv(p, index=False, encoding="utf-8-sig")
    print(f"    {p}")


# ================================================================= round 2
def build_round2(df_r1):
    print()
    print("=" * 70)
    print("ROUND 2 — test-set noise ceiling")
    print("=" * 70)

    for p in (TEST_LOCKED, MANIFEST):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}\nRun build_split_v2 first.")

    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)

    test = pd.read_csv(TEST_LOCKED, encoding="utf-8")
    test = test.loc[:, ~test.columns.str.contains("^Unnamed")]
    test["key"] = test["text"].map(norm)
    assert len(test) == man["n_test"], "test set differs from the manifest"
    assert sha256(TEST_LOCKED) == man["test_locked_sha256"], \
        "test set has changed since build_split_v2"
    print(f"  test set: {len(test)} rows, hash verified")

    # Exclude everything shown in round 1. The authoritative source is the key
    # of the batch actually handed to the annotators; the rebuild is unioned in
    # so a mismatch can only ever exclude more, never less.
    seen = set(df_r1["text"].map(norm))
    existing_key = f"{R1_DIR}/round1_key_PRIVATE.csv"
    if os.path.exists(existing_key):
        old_key = pd.read_csv(existing_key, encoding="utf-8-sig")
        extra = set(old_key["text"].map(norm)) - seen
        seen |= extra
        print(f"  round-1 texts from rebuild : {len(df_r1)}")
        print(f"  additional from the key    : {len(extra)}")
    print(f"  excluded as already shown  : {len(seen)}")

    distinct = test.drop_duplicates(subset="key")
    pool = distinct[~distinct["key"].isin(seen)].copy()
    print(f"  distinct texts in test     : {len(distinct)} "
          f"({len(test) - len(distinct)} duplicate rows collapsed)")
    print(f"  candidate pool             : {len(pool)} "
          f"({len(pool) / len(distinct):.1%} of the distinct test texts)")

    # contradictory-gold rows are kept: they are part of the noise being measured
    dev = pd.read_csv(DEV_POOL, encoding="utf-8")
    both = pd.concat([dev[["text", "label"]], test[["text", "label"]]])
    both["key"] = both["text"].map(norm)
    nconf = both.groupby("key")["label"].nunique()
    pool["gold_conflict"] = pool["key"].isin(set(nconf[nconf > 1].index))
    print(f"  with contradictory gold : {int(pool['gold_conflict'].sum())} "
          f"(kept, flagged in the key)")

    pool["stratum"] = pool["label"].astype(str) + "_" + pool["category"].astype(str)
    alloc = allocate_exact(pool["stratum"].value_counts(), N_CEILING)

    parts = []
    for stratum, n in alloc.items():
        if n <= 0:
            continue
        sub = pool[pool["stratum"] == stratum]
        parts.append(sub.sample(n=int(n), random_state=SEED_R2))
    cei = (pd.concat(parts)
           .sample(frac=1.0, random_state=SEED_R2)
           .reset_index(drop=True))

    assert len(cei) == N_CEILING, f"drew {len(cei)}, expected {N_CEILING}"
    assert cei["key"].is_unique, "duplicate text in round 2"
    assert not (seen & set(cei["key"])), "round 2 overlaps round 1"

    cei.insert(0, "id", [f"r2_{i:04d}" for i in range(len(cei))])
    cei["gold_name"] = cei["label"].map(NAME)

    for col, full in (("label", test["label"]), ("category", test["category"])):
        a = 100 * full.value_counts(normalize=True)
        b = 100 * cei[col].value_counts(normalize=True)
        comp = pd.DataFrame({"test_%": a, "sample_%": b}).fillna(0).round(2)
        comp["diff_pp"] = (comp["sample_%"] - comp["test_%"]).round(2)
        print(f"\n  {col} distribution:")
        print("   " + comp.to_string().replace("\n", "\n   "))

    return cei, test, pool, man, seen


def write_round2(cei):
    os.makedirs(R2_DIR, exist_ok=True)
    print()
    write_pair(cei, R2_DIR, "round2", SEED_R2 + 1, SEED_R2 + 2)
    key = cei[["id", "gold_name", "label", "category", "gold_conflict", "text"]].copy()
    key = key.rename(columns={"label": "gold_label"})
    key.insert(1, "arm", "test_ceiling")
    key.insert(2, "evaluated", True)
    p = f"{R2_DIR}/round2_key_PRIVATE.csv"
    key.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"    {p}")


# ===================================================================== main
def main():
    df_r1 = build_round1()
    ok = verify_round1(df_r1)

    if WRITE_ROUND1:
        if ok is False:
            sys.exit("refusing to overwrite: the rebuild does not match the "
                     "batch already annotated. Set WRITE_ROUND1 = False or "
                     "resolve the input versions first.")
        print("\n  writing round 1:")
        write_round1(df_r1)
    else:
        print("  WRITE_ROUND1 = False - round-1 files left untouched")

    cei, test, pool, man, seen = build_round2(df_r1)
    write_round2(cei)

    meta = {
        "round1": {
            "n_total": int(len(df_r1)),
            "n_evaluated": int(df_r1["evaluated"].sum()),
            "n_context": int(df_r1["context_only"].sum()),
            "seed": SEED_R1,
            "reproduces_existing": ok,
        },
        "round2": {
            "n": int(len(cei)),
            "seed": SEED_R2,
            "test_total": int(len(test)),
            "pool_after_exclusions": int(len(pool)),
            "n_excluded_as_seen": len(seen),
            "test_locked_sha256": man["test_locked_sha256"],
        },
        "inputs": {
            "KurdiSent.csv": sha256(ORIG),
            "KurdiSent_preprocessed.csv": sha256(PRE),
            "misclassified.csv": sha256(MIS),
        },
    }
    os.makedirs(ANNOTATION, exist_ok=True)
    p = f"{ANNOTATION}/annotation_build_meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print()
    print("=" * 70)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwritten: {p}")
    print("\nUpload to Label Studio: round1_import_annotatorB.csv, then the two "
          "round2 imports.\nEverything with PRIVATE in the name stays local.")


if __name__ == "__main__":
    main()
