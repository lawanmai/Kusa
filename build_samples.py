"""
KurdiSent re-annotation sampler (raw text throughout, seed=42).

Builds a single blind annotation stream from three arms:
  1. Representative sample  : 500 items, proportional to label x category
  2. News top-up            : 150 extra news items (disjoint from arm 1)
  3. Audit set              : confident model errors (conf >= 0.99)

All items — including the audit set — are emitted as RAW text. Audit items come
from the misclassified export in `surface` form; they are mapped back to raw text
via the row-aligned preprocessed file, then merged by raw text so nothing appears
twice. Outputs a blind import (id + text) and a private key (id -> gold, arm, ...).
"""
import pandas as pd, numpy as np, re, unicodedata

# ----------------------------- config -----------------------------
SEED           = 42
N_REPRESENT    = 500
N_NEWS_TOPUP   = 150
AUDIT_CONF_MIN = 0.99

ORIG = "/mnt/user-data/uploads/KurdiSent.csv"
PRE  = "/mnt/user-data/uploads/KurdiSent_preprocessed.csv"
MIS  = "/mnt/user-data/uploads/kusa_baseline_cv_inc_error_analysis_alternative_misclassified.csv"

NAME = {0: "Neutral", 1: "Negative", 2: "Positive"}
INV  = {v: k for k, v in NAME.items()}

def norm(s):
    """Normalize a string for matching: NFKC, drop ZWNJ + whitespace, unify variants."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u200c", "")
    s = re.sub(r"\s+", "", s)
    return s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")

# sampling frame
orig = pd.read_csv(ORIG, encoding="utf-8-sig")[["num_label", "category", "text"]].copy()
orig["text"] = orig["text"].astype(str).str.strip()

# drop contradictory gold labels
multi = orig.groupby("text")["num_label"].nunique()
conflict = set(multi[multi > 1].index)

frame = (orig[~orig["text"].isin(conflict)]
         .drop_duplicates("text")
         .reset_index(drop=True))
frame["key"] = frame["text"].map(norm)
frame_keys    = set(frame["key"])
conflict_keys = {norm(t) for t in conflict}   # normalized keys of the 8 excluded texts
print(f"Frame: {len(frame)} unique texts (excluded {len(conflict)} contradictory-label texts)")

# surface -> raw map (preprocessed is row-aligned)
pre = pd.read_csv(PRE, encoding="utf-8-sig")
pre["sk"] = pre["surface"].map(norm)
surf2raw = pre.drop_duplicates("sk").set_index("sk")["text"].str.strip()

# 1. representative (proportional)
cells = frame.groupby(["num_label", "category"]).size().rename("n").reset_index()
cells["exact"] = cells["n"] / cells["n"].sum() * N_REPRESENT
cells["base"]  = np.floor(cells["exact"]).astype(int)
remainder = N_REPRESENT - int(cells["base"].sum())
cells = cells.sort_values(cells["exact"].sub(cells["base"]).name if False else "exact",
                          ascending=False).reset_index(drop=True)
# largest-remainder
frac_order = (cells["exact"] - np.floor(cells["exact"])).sort_values(ascending=False).index
for i in frac_order[:remainder]:
    cells.loc[i, "base"] += 1
alloc = cells.set_index(["num_label", "category"])["base"].to_dict()

rep_idx = []
for (lab, cat), k in alloc.items():
    if k <= 0:
        continue
    pool = frame[(frame.num_label == lab) & (frame.category == cat)]
    rep_idx += pool.sample(n=min(k, len(pool)), random_state=SEED).index.tolist()
rep = frame.loc[rep_idx].copy()
print(f"Representative: {len(rep)} (news within it: {(rep.category=='news').sum()})")

# 2. news top-up
news_pool = frame[(frame.category == "news") & (~frame.index.isin(rep.index))]
topup = news_pool.sample(n=min(N_NEWS_TOPUP, len(news_pool)), random_state=SEED + 1).copy()
print(f"News top-up: {len(topup)} -> total news = {(rep.category=='news').sum() + len(topup)}")

# 3. audit set (mapped to RAW)
mis = pd.read_csv(MIS, encoding="utf-8-sig")
audit = mis[mis["confidence"] >= AUDIT_CONF_MIN].copy()
audit["sk"]  = audit["surface"].map(norm)
audit["raw"] = audit["sk"].map(surf2raw)            # surface -> raw text
n_unmapped = audit["raw"].isna().sum()
if n_unmapped:
    # fallback: keep surface text if no raw match (flagged)
    audit["raw"] = audit["raw"].fillna(audit["surface"].str.strip())
audit["gold_num"] = audit["true_name"].map(INV)
print(f"Audit: {len(audit)} items (conf>={AUDIT_CONF_MIN}); mapped to raw, {n_unmapped} fell back to surface")

# merge raw
items = {}  # norm(raw) -> record
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
n_audit_dropped = 0
for _, r in audit.iterrows():
    fk = norm(r["raw"])
    # exclude the 8 contradictory-label texts from every arm, including audit
    if fk in conflict_keys:
        n_audit_dropped += 1
        continue
    # prefer authoritative gold from the frame if this raw text is known
    if fk in frame_keys:
        fr = frame[frame.key == fk].iloc[0]
        add(fr.text, fr.num_label, fr.category, "audit", r.pred_name, float(r.confidence))
    else:
        add(r["raw"], r.gold_num, r.category, "audit", r.pred_name, float(r.confidence),
            textform=("surface" if pd.isna(surf2raw.get(r["sk"], np.nan)) else "raw"))
print(f"Audit dropped (contradictory-label texts): {n_audit_dropped}")

# assemble, shuffle, assign ids
rows = []
for v in items.values():
    rows.append(dict(text=v["text"], gold_label=v["gold"], gold_name=NAME[v["gold"]],
                     category=v["category"], arm=";".join(sorted(v["arm"])),
                     is_audit=int("audit" in v["arm"]), textform=v["textform"],
                     model_pred=v["model_pred"], model_conf=v["model_conf"]))
df = (pd.DataFrame(rows)
      .sample(frac=1.0, random_state=SEED + 2)
      .reset_index(drop=True))
df.insert(0, "id", [f"item_{i:04d}" for i in range(len(df))])

# write
df[["id", "text"]].to_csv("/home/claude/annotation_import_labelstudio.csv",
                          index=False, encoding="utf-8-sig")
key_cols = ["id", "gold_name", "gold_label", "category", "arm", "is_audit",
            "textform", "model_pred", "model_conf", "text"]
df[key_cols].to_csv("/home/claude/annotation_key_PRIVATE.csv",
                    index=False, encoding="utf-8-sig")

# report
dup_texts = df["text"].map(norm).duplicated().sum()
print("\n===== SUMMARY =====")
print("Total unique items:", len(df))
for a in ["representative", "news_topup", "audit"]:
    print(f"  contains '{a}': {df['arm'].str.contains(a).sum()}")
print("textform values :", df["textform"].value_counts().to_dict())
print("audit items      :", int((df.is_audit == 1).sum()))
print("duplicate texts  :", dup_texts)
print("news for analysis:",
      df[(df.category=='news') & (df.arm.str.contains('representative|news_topup'))].shape[0])
