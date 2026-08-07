# impute_equivalents.py
import re
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from src.impute_equivalent import impute_equivalents
from math import factorial as fact
from scipy.stats import ttest_ind


DIMS = ["safe","lively","beautiful","wealthy","depressing","boring"]
CATS = ["building","person","vegetation","street","traffic","sky"]
SEP = r"[+_]"  # treat '+' and '_' as equivalent

SCORE_COLS = [f"score_{c}" for c in DIMS]
RAT_COLS   = [f"rationale_{c}" for c in DIMS]
CONF_COLS  = [f"confidence_{c}" for c in DIMS]
RESULT_COLS = SCORE_COLS + RAT_COLS + CONF_COLS

def label_tokens(label: str) -> tuple[str, ...]:
    parts = [p.strip().lower() for p in re.split(SEP, str(label)) if p.strip()]
    toks = [p for p in parts if p in DIMS]
    return tuple(sorted(toks))

def toks_key(toks: tuple[str, ...]) -> str:
    return "+".join(toks)  # canonical string key

def label_tokens(label: str):
    parts = [p.strip().lower() for p in re.split(SEP, str(label)) if p.strip()]
    toks = [p for p in parts if p in CATS]
    return tuple(sorted(toks))

def build_f_values_for_image(df_img: pd.DataFrame, score_col: str) -> dict:
    """
    Map frozenset(visible_CATS) -> score for this image & dimension.
    Assumes df_img has one row per label (64 total after imputation).
    """
    fvals = {}
    for _, row in df_img.iterrows():
        B = set(label_tokens(row["label"]))   # blurred categories
        S = frozenset(set(CATS) - B)          # visible categories
        fvals[S] = float(row[score_col])
    return fvals

def shapley_from_fvalues(fvals: dict) -> dict:
    """Exact Shapley values given complete f(S) over all S ⊆ V."""
    V = tuple(CATS)
    m = len(V)
    # quick access
    def f(S): return fvals[frozenset(S)]

    phi = {i: 0.0 for i in V}
    for i in V:
        V_minus_i = [v for v in V if v != i]
        for r in range(0, m):  # subsets S of V\{i}
            for S in combinations(V_minus_i, r):
                w = (fact(r) * fact(m - r - 1)) / fact(m)
                phi[i] += w * ( f(S + (i,)) - f(S) )
    return phi


def build_props_long(props: pd.DataFrame) -> pd.DataFrame:
    # find the best column for each category
    suffixes = ["_p", "_area_full", "_area", "_prop", "_pct"]
    col_for = {}
    lower_cols = {c.lower(): c for c in props.columns}
    for cat in CATS:
        for s in suffixes:
            key = f"{cat}{s}"
            if key in lower_cols:
                col_for[cat] = lower_cols[key]
                break
        if cat not in col_for:
            raise KeyError(f"No proportion column found for '{cat}'. "
                           f"Looked for: {', '.join(cat+s for s in suffixes)}")
    # keep and rename to category names
    keep = ["image_id"] + [col_for[c] for c in CATS]
    tmp = props[keep].rename(columns={col_for[c]: c for c in CATS})
    # long format: image_id, feature, p
    return tmp.melt(id_vars="image_id", var_name="feature", value_name="p")

def build_phi(df, y='score', seq_id=None):
    all_phi = []  # rows: image_id x dimension x feature φ
    for img_id, g in df.groupby("image_id"):
        for dim in DIMS:
            score_col = f"{y}_{dim}"
            # skip if this image/dimension is missing any scores
            if g[score_col].isna().any():
                print(img_id)
                continue
            fvals = build_f_values_for_image(g, score_col)
            # sanity: must have all 2^m keys
            if len(fvals) != (1 << len(CATS)):
                # if you truly imputed to all 64, this shouldn't happen
                continue
            phi = shapley_from_fvalues(fvals)
            for feat, val in phi.items():
                conf = g[g.label=='unmasked'][f'confidence_{dim}'].iloc[0]
                all_phi.append({"image_id": img_id, "dimension": dim, "feature": feat, "phi": val, "confidence":int(conf)})

    phi_df = pd.DataFrame(all_phi)
    phi_df = phi_df.sort_values(by='image_id')
    phi_df = add_m_pc(phi_df, df)
    phi_df['seq_id'] = seq_id
    return phi_df

def add_m_pc(phi_df, results_df):
    """Adds each m_pc from results_df to the appropriate row in phi_df"""
    m_col = []
    unmasked_df = results_df[results_df.label=='unmasked']
    for i, row in phi_df.iterrows():
        img_id = row.image_id
        p = row.dimension
        c = row.feature
        img_df = unmasked_df[unmasked_df.image_id==img_id]
        m_col.append(img_df[f'{p}_{c}_m'].iloc[0])
    phi_df['m_pc'] = m_col
    phi_df['m_pc'] = phi_df['m_pc'].replace(np.nan, 'U')
    return phi_df

def pivot_matrix(phi_df: pd.DataFrame, image_order=None):
    # pivot to rows=image, columns=(dimension, feature)
    piv = (phi_df
           .pivot_table(index="image_id",
                        columns=["dimension","feature"],
                        values="phi", aggfunc="mean"))
    # reindex columns to fixed order DIMS x CATS
    piv = piv.sort_index()
    piv = piv.reindex(columns=pd.MultiIndex.from_product([DIMS, CATS]))
    return piv


def build_delta_phi(full_df):
    results = []
    for feat in CATS:
        for dim in DIMS:
            sub = full_df[(full_df["feature"]==feat) &
                        (full_df["dimension"]==dim) &
                        (full_df["m_pc"].isin(["PP","NP","U"]))]

            groups = {k: sub.loc[sub["m_pc"]==k, "phi"].dropna().to_numpy()
                    for k in ["PP","NP","U"]}

            for g1, g2 in [("PP","NP"), ("PP","U"), ("NP","U")]:
                if len(groups[g1])>1 and len(groups[g2])>1:
                    stat, pval = ttest_ind(groups[g1], groups[g2], equal_var=False)
                    results.append({
                        "feature": feat,
                        "dimension": dim,
                        "comparison": f"{g1} vs {g2}",
                        "n1": len(groups[g1]),
                        "n2": len(groups[g2]),
                        "t_stat": stat,
                        "pval": pval,
                        "mean1":np.mean(groups[g1]),
                        "mean2":np.mean(groups[g2]),
                        "significant":pval<0.06
                    })

    results_df = pd.DataFrame(results)
    return results_df
