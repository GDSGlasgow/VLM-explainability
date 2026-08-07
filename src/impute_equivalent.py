# impute_equivalents.py
import re
import pandas as pd
import numpy as np
from pathlib import Path

CHAR_ORDER = ["safe","lively","beautiful","wealthy","depressing","boring"]
CATEGORIES = ["building","person","vegetation","street","traffic","sky"]
SEP_PATTERN = r"[+_]"  # treat '+' and '_' as equivalent

SCORE_COLS = [f"score_{c}" for c in CHAR_ORDER]
RAT_COLS   = [f"rationale_{c}" for c in CHAR_ORDER]
CONF_COLS  = [f"confidence_{c}" for c in CHAR_ORDER]
RESULT_COLS = SCORE_COLS + RAT_COLS + CONF_COLS

def label_tokens(label: str) -> tuple[str, ...]:
    parts = [p.strip().lower() for p in re.split(SEP_PATTERN, str(label)) if p.strip()]
    toks = [p for p in parts if p in CATEGORIES]
    return tuple(sorted(toks))

def toks_key(toks: tuple[str, ...]) -> str:
    return "+".join(toks)  # canonical string key

def impute_equivalents(df:pd.DataFrame, min_frac: float = 0.0) -> None:
    """
    For each image_id:
      1) Determine 'present' categories (max(<cat>_p) > min_frac)
      2) Compute effective token set for every row: tokens(label) ∩ present
      3) Find a canonical row per effective set that already has VLM outputs
      4) Copy those outputs into rows with missing values for the same effective set
    """
    if "image_id" not in df or "label" not in df:
        raise ValueError("Expected columns 'image_id' and 'label' in results CSV")

    # Normalize
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(str).str.lower()

    # Ensure all result columns exist (so combine_first works cleanly)
    for c in RESULT_COLS:
        if c not in df.columns:
            if c.startswith(("score_", "confidence_")):
                df[c] = np.nan
            else:
                df[c] = ""

    # Tokenize labels once
    df["tokens"] = df["label"].apply(label_tokens)

    # Compute per-image present categories using metadata columns "<cat>_p"
    present_map: dict[str, set[str]] = {}
    for img_id, g in df.groupby("image_id"):
        present = set()
        for c in CATEGORIES:
            col = f"{c}_p"
            if col in g.columns:
                mx = pd.to_numeric(g[col], errors="coerce").max(skipna=True)
                if pd.notna(mx) and mx > float(min_frac):
                    present.add(c)
        present_map[img_id] = present

    # Effective token set = tokens ∩ present
    def eff_tokens_row(row):
        pres = present_map.get(row["image_id"], set())
        return tuple(sorted([t for t in row["tokens"] if t in pres]))
    df["eff_tokens"] = df.apply(eff_tokens_row, axis=1)
    df["eff_key"] = df["eff_tokens"].apply(toks_key)

    # Build canonical donors per (image_id, eff_key): pick the row with the most non-null scores
    donors = []
    for img_id, g in df.groupby("image_id"):
        # rows that already have some VLM output
        nonnull_counts = g[SCORE_COLS].notna().sum(axis=1)
        g2 = g.assign(_nn=nonnull_counts)
        # keep the best donor per eff_key (max non-null count; tie-breaker: prefer label == eff_key if present)
        g2 = g2.sort_values(by=["eff_key", "_nn"], ascending=[True, False])
        # optional tie-break: if label with exact canonical ordering exists, prefer it
        g2["_is_exact_eff"] = (g2["label"].apply(label_tokens).apply(toks_key) == g2["eff_key"]).astype(int)
        g2 = g2.sort_values(by=["eff_key", "_is_exact_eff", "_nn"], ascending=[True, False, False])
        donors.append(
            g2.drop_duplicates(subset=["eff_key"], keep="first")[
                ["image_id","eff_key"] + RESULT_COLS
            ]
        )
    canon = pd.concat(donors, ignore_index=True)

    # Merge donors back; fill missing from canonical equivalents
    merged = df.merge(canon, on=["image_id","eff_key"], how="left", suffixes=("", "__canon"))
    for c in RESULT_COLS:
        merged[c] = merged[c].combine_first(merged.get(f"{c}__canon"))
    drop_cols = [c for c in merged.columns if c.endswith("__canon")]
    merged.drop(columns=drop_cols, inplace=True)

    # (Optional) report fill stats
    before_missing = df[SCORE_COLS].isna().sum().sum()
    after_missing  = merged[SCORE_COLS].isna().sum().sum()
    print(f"Imputed scores: filled {before_missing - after_missing} missing entries "
          f"(remaining {after_missing}).")

    return merged
if __name__ == "__main__":
    # Example:
    # impute_equivalents('results/myseq.csv', 'results/myseq_imputed.csv', min_frac=0.0)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to results CSV.")
    ap.add_argument("--out", dest="out", required=True, help="Path to write imputed CSV.")
    ap.add_argument("--min-frac", type=float, default=0.0, help="Presence threshold for categories.")
    args = ap.parse_args()
    impute_equivalents(args.inp, args.out, min_frac=args.min_frac)