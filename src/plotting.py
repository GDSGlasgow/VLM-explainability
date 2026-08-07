import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.impute_equivalent import impute_equivalents
from scipy.stats import spearmanr, kendalltau, ttest_1samp
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch


DIMS = ["safe","lively","beautiful","wealthy","depressing","boring"]
CATS = ["building","person","vegetation","street","traffic","sky"]

def blue_white_pink():
    # custom diverging cmap: pink -> white -> blue
    return LinearSegmentedColormap.from_list(
        "pwb", [ "#3B4EB6", "#ffffff", "#F14272"], N=256
    )


def choose_least_busy_corner(x, y, ax, frac_w=0.22, frac_h=0.18, margin=0.02):
    """
    Pick TL/TR/BL/BR by counting how many points fall under a box
    that would occupy `frac_w` × `frac_h` of the axes (plus a small margin).
    Returns: (u, v, ha, va, corner_tag)
      where (u,v) are axes-fraction coords for ax.text, and ha/va are alignments.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return 0.98, 0.02, "right", "bottom", "br"

    x, y = x[m], y[m]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    W, H = (x1 - x0), (y1 - y0)
    pad_x, pad_y = margin * W, margin * H
    bw, bh = frac_w * W, frac_h * H

    rects = {
        "br": (x1 - bw - pad_x, x1 - pad_x,      y0 + pad_y,       y0 + bh + pad_y),
        "tr": (x1 - bw - pad_x, x1 - pad_x,      y1 - bh - pad_y,  y1 - pad_y),
        "bl": (x0 + pad_x,      x0 + bw + pad_x, y0 + pad_y,       y0 + bh + pad_y),
        "tl": (x0 + pad_x,      x0 + bw + pad_x, y1 - bh - pad_y,  y1 - pad_y),
    }

    counts = {}
    for tag, (xa, xb, ya, yb) in rects.items():
        counts[tag] = int(((x >= xa) & (x <= xb) & (y >= ya) & (y <= yb)).sum())

    best = min(counts, key=counts.get)
    # map to axes-fraction coords + alignments
    corner_map = {
        "br": (0.98, 0.02, "right", "bottom"),
        "tr": (0.98, 0.98, "right", "top"),
        "bl": (0.02, 0.02, "left",  "bottom"),
        "tl": (0.02, 0.98, "left",  "top"),
    }
    u, v, ha, va = corner_map[best]
    return u, v, ha, va, best

# Update the function to make the legend single-column and stretch the x-axis a bit.
def plot_sequence_stack(metadata_path, category_order=None, ax=None, save_path=None, cmap='Tab2', legend=True):
    """
    Plot a stacked bar chart showing, for each image in a sequence, the proportion of
    each segmentation category (from <category>_p columns). For each image_id, the max
    value per category is used.
    """
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)
    df['image_id'] = df['image_id'].astype(int)
    df = df.sort_values('image_id')
    cat_cols = [c for c in df.columns if c.endswith("_p")]
    if not cat_cols:
        raise ValueError("No '<category>_p' columns found in the metadata.")
    if "image_id" not in df.columns:
        raise ValueError("Expected an 'image_id' column in the metadata.")
    
    agg_dict = {c: "max" for c in cat_cols}
    grouped = df.groupby("image_id", as_index=False).agg(agg_dict)
    
    cats = [c[:-2] for c in cat_cols]
    seen = set()
    cats_unique = [c for c in cats if not (c in seen or seen.add(c))]
    
    if category_order is None:
        preferred = ["person", "building", "traffic", "vegetation"]
    else:
        preferred = list(category_order)
    preferred = [c for c in preferred if f"{c}_p" in cat_cols]
    remaining = sorted([c for c in cats_unique if c not in preferred])
    final_order = preferred + remaining
    
    ordered_cols = ["image_id"] + [f"{c}_p" for c in final_order]
    ordered_cols = [c for c in ordered_cols if c in grouped.columns]
    grouped = grouped[ordered_cols]
    
    value_cols = [c for c in grouped.columns if c != "image_id"]
    values = grouped[value_cols].to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        values_norm = np.where(row_sums > 0, values / row_sums, values)
    grouped[value_cols] = values_norm
    
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 3.2))  # stretched x axis by 1.2x (was 10 -> 12)
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure
    
    x = np.arange(len(grouped))
    width = 1.0
    bottoms = np.zeros(len(grouped))
    num_cats = len(value_cols)
    # set colours
    if cmap is None:
        base_cmap = plt.get_cmap("tab20")
        colors = [base_cmap(i % 20) for i in range(num_cats)]
    else:
        if isinstance(cmap, list):
            colors = [cmap[i % len(cmap)] for i in range(num_cats)]
        else:
            base_cmap = plt.get_cmap(cmap)
            colors = [base_cmap(i % 20) for i in range(num_cats)]
    
    for idx, col in enumerate(value_cols):
        ax.bar(x, grouped[col].to_numpy(), bottom=bottoms, width=width, align="edge", label=col[:-2], color=colors[idx])
        bottoms += grouped[col].to_numpy()
        
    for i in range(1, len(grouped)):
        ax.axvline(i, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(grouped))
    ax.margins(x=0)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Proportion of image")
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([str(i) for i in df.image_id.unique()], rotation=0)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    
    # Legend: single column, placed to the right
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False, ncol=1, title="Category")
    
    plt.tight_layout(pad=0.5)
    
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    return fig, ax, grouped



def plot_unmasked_placepulse_heatmap(csv_path, dims=None, order_col=None, vmin=0, vmax=100, save_path=None, ax=None, show_values=False, cmap=None, add_colorbar=True):
    """
    Heatmap (6 x N) for unmasked images across PlacePulse dimensions.
    Rows: dimensions; Columns: image order (unmasked only).
    
    Parameters
    ----------
    csv_path : str/Path
        Path to results CSV containing a 'label' column with 'unmasked' rows.
    dims : list[str] or None
        Optional specific dimension names. Accepts raw names (e.g., 'safe') or 'score_safe'.
        If None, auto-detects columns beginning with 'score_'.
    order_col : str or None
        Column to sort by (e.g., 'frame_idx' or 'image_id'). If None, uses 'frame_idx' then 'image_id' when available.
    vmin, vmax : float
        Value range for the heatmap colour scale.
    save_path : str/Path or None
        If provided, saves the figure to this path.
    ax : matplotlib.axes.Axes or None
        If provided, plot into this axes.
    show_values : bool
        If True, overlay the numeric score in each cell.
        
    Returns
    -------
    fig, ax, used_dims, df_unmasked_sorted
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df['image_id'] = df['image_id'].astype(int)
    df = df.sort_values('image_id')
    df = impute_equivalents(df, min_frac=0.01)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column with 'unmasked' entries.")
    df_unmasked = df[df["label"].astype(str).str.lower() == "unmasked"].copy()
    if df_unmasked.empty:
        raise ValueError("No rows with label == 'unmasked' were found.")
    df_unmasked['image_id'] = df_unmasked['image_id'].astype('int')
    df_unmasked = df_unmasked.sort_values('image_id')
    # Determine dimension columns
    score_cols = [c for c in df_unmasked.columns if str(c).startswith("score_")]
    if dims is None:
        dims_used = score_cols
    else:
        dims_used = []
        dims_set = set(dims) | {f"score_{d}" for d in dims}
        for c in score_cols:
            if c in dims_set:
                dims_used.append(c)
    if len(dims_used) == 0:
        raise ValueError("No dimension columns found. Expected columns like 'score_safe', 'score_lively', etc.")
    
    # Order rows (dimensions) in a stable friendly order if possible
    pref = ["score_safe","score_lively","score_beautiful","score_wealthy","score_depressing","score_boring"]
    dims_used = [c for c in pref if c in dims_used] + [c for c in dims_used if c not in pref]
    
    # Decide ordering for columns (images)
    if order_col is None:
        if "frame_idx" in df_unmasked.columns:
            order_col = "frame_idx"
        elif "image_id" in df_unmasked.columns:
            order_col = "image_id"
    if order_col is not None:
        df_unmasked = df_unmasked.sort_values(order_col)
    #df_unmasked = df_unmasked.reset_index(drop=True)
    df_unmasked = df_unmasked.drop_duplicates(subset='image_id', keep='first')
    # Build matrix: rows = dims, cols = images
    data = np.vstack([df_unmasked[c].to_numpy() for c in dims_used])
    
    # Create axes if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 3.6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        
        # Build matrix
    data = np.vstack([df_unmasked[c].to_numpy() for c in dims_used])

    # Handle cmap option
    if cmap is None:
        cmap_used = "viridis"
    elif isinstance(cmap, list):
        cmap_used = mcolors.LinearSegmentedColormap.from_list("custom", cmap)
    else:
        cmap_used = cmap

    # Plot
    im = ax.imshow(data, aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax, cmap=cmap_used)
    
    #im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    
    # Axis ticks: integers for images, labels for dims
    n_imgs = data.shape[1]
    ax.set_xticks(np.arange(n_imgs))
    ax.set_xticklabels([int(i) for i in df_unmasked.image_id])
    ax.set_yticks(np.arange(len(dims_used)))
    ax.set_yticklabels([d.replace("score_","") for d in dims_used])
    
    # Minor gridlines around cells (default styling)
    ax.set_xticks(np.arange(-0.5, n_imgs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dims_used), 1), minor=True)
    ax.grid(which="minor")
    
    ax.set_xlabel("Image index")
    #ax.set_ylabel("PlacePulse dimension")
    
    # Optional numeric overlays
    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{int(round(data[i, j]))}", ha="center", va="center")
    
    # Colorbar
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Score")
    
    plt.tight_layout(pad=0.5)
    
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    return fig, ax, [d.replace("score_","") for d in dims_used], df_unmasked


def binned_median_trend(x, y, nbins=4):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < max(10, nbins): return None
    # equal-count bins
    q = np.linspace(0, 1, num=nbins+1)
    edges = np.unique(np.quantile(x, q))
    if len(edges) < 3: return None
    xm, ym, yq1, yq3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x <= b) if b == edges[-1] else (x >= a) & (x < b)
        if sel.sum() == 0: continue
        xm.append(np.median(x[sel]))
        ys = y[sel]
        ym.append(np.median(ys))
        yq1.append(np.percentile(ys, 25))
        yq3.append(np.percentile(ys, 75))
    return np.array(xm), np.array(ym), np.array(yq1), np.array(yq3)

def panel(df_in, ycol="phi", xcol="p", title="φ vs proportion",
          nbins=8, jitter_frac=0.01, figsize=(12,12),
          rho_sig_thresh=0.44, base_font=10, nrows=6, ncols=6):
    """
    Plots 6x4 grid. ρ is bottom-right in a white box; red if |ρ| >= rho_sig_thresh.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    stats = []

    for i, dim in enumerate(DIMS):
        for j, cat in enumerate(CATS):
            ax = axes[i, j]
            sub = df_in[(df_in.dimension==dim) & (df_in.feature==cat)]

            x = sub[xcol].to_numpy(float)
            y = sub[ycol].to_numpy(float)
            sgn = np.sign(sub["phi"].to_numpy(float))

            # optional small jitter to reduce overlap
            if jitter_frac > 0:
                span = (np.nanmax(x) - np.nanmin(x)) or 1.0
                x = np.clip(x + rng.uniform(-jitter_frac, jitter_frac, size=len(x))*span, 0, 1)

            # scatter by sign
            pos = sgn >= 0
            ax.scatter(x[pos],  y[pos],  s=40, c=PINK, ec="none", label="φ>0")
            ax.scatter(x[~pos], y[~pos], s=40, c=BLUE, ec="none", label="φ<0")

            # robust median + IQR (neutral)
            tr = binned_median_trend(x, y, nbins=nbins)
            if tr:
                xm, ym, yq1, yq3 = tr
                ax.plot(xm, ym, color=NEUTRAL, lw=1.8, alpha=0.9)
                ax.vlines(xm, yq1, yq3, color=NEUTRAL, lw=1.2, alpha=0.5)
            else:
                print(tr)

            # Spearman ρ (signed), show bottom-right in a white box
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 4:
                rho, pval = spearmanr(x[m], y[m])
                tau, tau_pval = kendalltau(x[m], y[m], variant='b')
                stats.append({"dimension": dim, "feature": cat, "rho": rho, "pval": pval, "tau":tau, "tau_pval":tau_pval, "ycol": ycol})
                sig = (abs(rho) >= rho_sig_thresh)
                color = "#c00000" if sig else "#333333"

                # choose corner after plotting points/lines so limits are final
                u, v, ha, va, which = choose_least_busy_corner(x[m], y[m], ax, frac_w=0.22, frac_h=0.18, margin=0.02)

                ax.text(u, v, f"ρ={rho:+.2f}",
                        transform=ax.transAxes, ha=ha, va=va,
                        fontsize=base_font+2, color=color,
                        bbox=dict(facecolor="white",
                                edgecolor=color,
                                boxstyle="round,pad=0.18", alpha=0.9))

            # labels & ticks a touch larger
            if i == 5: ax.set_xlabel(f"{cat} proportion" if xcol=="p" else cat, fontsize=base_font+1)
            if j == 0: ax.set_ylabel(f"{dim} ({ycol})", fontsize=base_font+1)
            ax.tick_params(labelsize=base_font)
            ax.grid(alpha=0.15)

    # FDR per metric
    stats = pd.DataFrame(stats)
    if not stats.empty:
        stats["qval"] = multipletests(stats["pval"], method="fdr_bh")[1]

    fig.suptitle(title, y=0.995, fontsize=base_font+3)
    fig.tight_layout()
    return fig, axes, stats

def plot_big_heatmap(piv: pd.DataFrame,
    out_png: str | None = None,
    title: str | None = None,
    max_xticks: int = 40,
    right_gutter_ratio: float = 1/6,   # ~16.7% gutter for right-side y labels
    cbar_ratio: float = 0.045,          # width fraction for colorbar column
    vmax: float | None = None,
    vmin: float | None = None,
    annotate: bool = False,
    annotate_threshold: float = 0.05,   # only label if |Δ| > this
    annotate_fmt: str = "{:.2f}",
    annotate_fontsize: int = 6,
    annotate_dynamic_color: bool = True
):
    """
    Rotated heatmap:
      rows  = (dimension, feature) blocks (6×6)
      cols  = images
      y-ticks on the RIGHT = category names, repeated per dimension block
      faint per-category lines, thicker per-dimension separators
      colorbar placed beyond a right gutter to avoid overlap with labels
    """
    #piv = piv.sort_index()
    mat = piv.T.values  # (6*6, n_images)
    n_rows, n_cols = mat.shape

    # robust symmetric color limits
    if not vmax:
        vmax = np.nanpercentile(np.abs(mat), 99)
        vmin = -vmax
    else:
        vmax = max(abs(vmax), abs(vmin)) if vmin is not None else abs(vmax)
        vmin = -vmax if vmin is None else min(-abs(vmax), abs(vmin))
    
    cmap = blue_white_pink()

    # figure size scales with matrix size
    fig_w = 0.15 * n_cols + 4
    fig_h = 0.25 * n_rows + 2

    # build a 1x3 grid: [heatmap | right-gutter | colorbar]
    # gutter ~ right_gutter_ratio of heatmap width; cbar ~ cbar_ratio
    heatmap_w = 1.0
    gutter_w  = max(0.12, right_gutter_ratio) * heatmap_w
    cbar_w    = cbar_ratio * heatmap_w

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(fig_w, fig_h), dpi=160,
        gridspec_kw={"width_ratios": [heatmap_w, gutter_w, cbar_w], "wspace": 0.02},
        constrained_layout=False
    )
    ax, ax_gutter, ax_cbar = axes
    ax_gutter.axis("off")  # pure spacing column

    im = ax.imshow(
        mat, cmap=cmap, vmin=-vmax, vmax=vmax,
        aspect="auto", interpolation="nearest", origin="upper"
    )
    if annotate:
        def _txt_color(val):
            if not annotate_dynamic_color:
                return "black"
            # white text on strong colors
            return "white" if abs(val) > 0.45 * vmax else "black"

        for r in range(n_rows):
            for c in range(n_cols):
                val = mat[r, c]
                if np.isfinite(val) and abs(val) > annotate_threshold:
                    ax.text(c, r, annotate_fmt.format(val),
                            ha="center", va="center",
                            fontsize=annotate_fontsize,
                            color=_txt_color(val))

    # --- X (images) ---
    ax.set_xlabel("Image index", fontsize=10)
    ax.set_xlim(-0.5, n_cols - 0.5)
    step = max(1, n_cols // max_xticks)
    xticks = np.arange(0, n_cols, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([i for i in piv.index], rotation=90, fontsize=7)

    # --- Y (36 rows = 6 dims × 6 cats) ---
    ytick_pos = np.arange(n_rows)
    ytick_lab = (CATS * len(DIMS))  # repeat categories per dimension block
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=8)

    # move y ticks/labels to the RIGHT (into the gutter space)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", which="both", labelleft=False, labelright=True, right=True, pad=8)

    # --- horizontal grid lines ---
    # faint lines separating categories (each row)
    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color="k", lw=0.4, alpha=0.18, zorder=2)
        
    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color="k", lw=0.4, alpha=0.18, zorder=2)

    # thicker separators between dimension blocks (every 6 rows)
    for g in range(1, len(DIMS)):
        y = g * len(CATS) - 0.5
        ax.axhline(y, color="k", lw=1.2, alpha=0.45, zorder=3)

    # dimension labels on the LEFT (won't collide with right ticks)
    for g, dim in enumerate(DIMS):
        y0 = g * len(CATS)
        y1 = y0 + len(CATS) - 1
        y_center = (y0 + y1) / 2.0
        ax.text(-1.2, y_center, dim, ha="right", va="center",
                fontsize=9, fontweight="bold", rotation=90, clip_on=False)

    # colorbar placed in the dedicated rightmost axes
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("$\phi_{p,c}(I)$", fontsize=12)

    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    return fig, (ax, ax_cbar)


def plot_phi_significance(
    sig_df: pd.DataFrame,
    comparison: tuple[str, str] | None = None,    # e.g. ("PP","U")
    *,
    dims=None,                                     # optional explicit order of columns
    features=None,                                 # optional explicit order of rows
    alpha: float = 0.05,
    n_min: int = 5,
    pcol: str = "pval",                            # or "p_fdr"
    cmap_red: str = "#F14272",
    cmap_blue: str = "#3B4EB6",
    nan_color: str = "#D8D8D8",
    vmin: float | None = None,
    vmax: float | None = None,
    ax=None,
    annotate: bool = True,
    fontsize=11
):
    """
    Draw one panel of Δφ = mean1 - mean2 over (feature, dimension).

    Returns (im, ax), where `im` is the AxesImage (useful for a shared colorbar).
    Does NOT make a colorbar or legend (handle those at the figure level).
    """

    # Normalise text cols
    df = sig_df.copy()
    df["feature"]   = df["feature"].astype(str).str.strip().str.lower()
    df["dimension"] = df["dimension"].astype(str).str.strip().str.lower()

    # Resolve row/column orders
    if features is None:
        features = CATS
    else:
        features = [str(f).strip().lower() for f in features]

    if dims is None:
        dims = DIMS
    else:
        dims = [str(d).strip().lower() for d in dims]

    # Build grids
    mean_diff = np.full((len(features), len(dims)), np.nan, dtype=float)
    sig_grid  = np.zeros((len(features), len(dims)), dtype=bool)
    n_grid = np.zeros((len(features), len(dims)), dtype=bool)
    for r, f in enumerate(features):
        for c, d in enumerate(dims):
            row = df[(df["feature"] == f) & (df["dimension"] == d)]

            if len(row):
                # store whether n < n_min
                if (row.n1.iloc[0] >= n_min) and (row.n2.iloc[0] >= n_min):
                    n_grid[r, c] = 1
                # store means an pvals
                m1 = float(row["mean1"].iloc[0])
                m2 = float(row["mean2"].iloc[0])
                pv = float(row[pcol].iloc[0])
                mean_diff[r, c] = m1 - m2
                sig_grid[r, c]  = (pv < alpha)

    # Colormap (blue → white → red) and NaN color
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", [cmap_blue, "white", cmap_red], N=256
    ).copy()
    cmap.set_bad(nan_color)

    # Symmetric scaling (unless user passes vmin/vmax)
    mdata = np.ma.masked_invalid(mean_diff)
    if vmax is None:
        finite = np.abs(mdata.compressed()) if mdata.count() > 0 else np.array([1.0])
        vmax = max(np.percentile(finite, 95), 1e-6)
    if vmin is None:
        vmin = -vmax

    # Axes
    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    else:
        fig = ax.figure

    im = ax.imshow(mdata, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Ticks/labels
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=fontsize)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=fontsize)

    # Faint grid
    ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Title per comparison
    def title_for(comp):
        if comp == ('PP','NP'):
            return r"(a) $\Delta_{p,c}(\mathrm{PP},\mathrm{NP})$"
        if comp == ('PP','U'):
            return r"(b) $\Delta_{p,c}(\mathrm{PP},U)$"
        if comp == ('NP','U'):
            return r"(c) $\Delta_{p,c}(\mathrm{NP},U)$"
        return r"$\Delta_{p,c}$"

    ax.set_title(title_for(comparison), fontsize=fontsize+3, pad=6)

    # Cell annotations with auto-contrast
    if annotate:
        thr = 0.6 * vmax if vmax else 0.0
        for r in range(len(features)):
            for c in range(len(dims)):
                val = mean_diff[r, c]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > thr else "black"
                    txt_weight = "bold" if n_grid[r, c]==1 else "normal"
                    if abs(val) >= 10:
                        s = f"{val:.1f}"
                    elif abs(val) >= 1:
                        s = f"{val:.2f}"
                    else:
                        s = f"{val:.3f}"
                    ax.text(c, r, s, ha="center", va="center", color=txt_color, 
                            fontsize=fontsize, weight=txt_weight)

    # Significance outlines
    for r in range(len(features)):
        for c in range(len(dims)):
            if sig_grid[r, c]:
                ax.add_patch(Rectangle((c-0.5, r-0.5), 1, 1,
                                       fill=False, edgecolor="#000000",
                                       linewidth=2.0, zorder=3))

    return im, ax


def phi_legend_handles(cmap_red="#F14272", cmap_blue="#3B4EB6",
                       nan_color="#D8D8D8", alpha=0.05, p_label="p"):
    """Figure-level legend handles."""
    return [
        Patch(facecolor=cmap_red,  edgecolor=cmap_red,
              label=r"$\bar{\phi}_{p,c}^{A} > \bar{\phi}_{p,c}^{B}$"),
        Patch(facecolor=cmap_blue, edgecolor=cmap_blue,
              label=r"$\bar{\phi}_{p,c}^{A} < \bar{\phi}_{p,c}^{B}$"),
        Patch(facecolor="none", edgecolor="#000000", linewidth=2.0,
              label=fr"{p_label}-value < {alpha:g}"),
        Patch(facecolor=nan_color, edgecolor=nan_color, label="no data"),
    ]
    
# class thresholds (you can tweak globally)
TH_SMALL  = 0.2   # |z| ≤ 0.2 for U
TH_MEDIUM = 0.5   # ≥0.5 for PP, ≤-0.5 for NP

def threshold_sig_mask(z_tab: pd.DataFrame, klass: str,
                       th_small=TH_SMALL, th_medium=TH_MEDIUM) -> pd.DataFrame:
    z = z_tab.copy()
    if klass == "PP":
        mask = z >= th_medium
    elif klass == "NP":
        mask = z <= -th_medium
    elif klass == "U":
        mask = z.abs() <= th_small
    else:
        raise ValueError("klass must be one of {'PP','NP','U'}")
    return mask.fillna(False)

def plot_z_heatmap_textsig(
    z_tab: pd.DataFrame,
    n_img: pd.DataFrame | None,
    klass: str,                               # "PP", "NP", or "U"
    *,
    features=None,                             # optional subset; ordered by global CATS if present
    dims=None,                                 # optional subset; ordered by global DIMS if present
    cmap_blue="#3B4EB6",
    cmap_red="#F14272",
    nan_color="#D8D8D8",
    vmin=None, vmax=None,                      # if None, robust symmetric ±p95 within this tab
    ax=None,
    text_fmt="{:.2f}",
    fontsize=11,
    low_n: int | None = 5,                     # if not None, append '*' when n < low_n
    annotate=True,
):
    
    if not features:
        features = CATS
    if not dims:
        dims = DIMS 

    z_tab = z_tab.reindex(index=features, columns=dims)
    if n_img is not None:
        n_img = n_img.reindex(index=features, columns=dims)

    # Build significance by class-threshold rule
    sig_tab = threshold_sig_mask(z_tab, klass)

    # Colormap & limits
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", [cmap_blue, "white", cmap_red], N=256
    ).copy()
    cmap.set_bad(nan_color)
    data = z_tab.to_numpy(dtype=float)
    mdata = np.ma.masked_invalid(data)

    if vmax is None:
        finite = np.abs(mdata.compressed()) if mdata.count() > 0 else np.array([1.0])
        vmax = max(np.percentile(finite, 95), 1e-6)
    if vmin is None:
        vmin = -vmax

    # Axes
    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    else:
        fig = ax.figure

    im = ax.imshow(mdata, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Ticks/labels
    ax.set_yticks(np.arange(len(features))); ax.set_yticklabels(features, fontsize=fontsize)
    ax.set_xticks(np.arange(len(dims)));     ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=fontsize)

    # Faint grid
    ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotations: z and n; bold if passes class threshold
    if annotate:
        for r, feat in enumerate(features):
            for c, dim in enumerate(dims):
                zval = z_tab.loc[feat, dim]
                if pd.isna(zval):
                    continue
                nval = int(n_img.loc[feat, dim]) if (n_img is not None and pd.notna(n_img.loc[feat, dim])) else 0
                significant = bool(sig_tab.loc[feat, dim])
                # auto-contrast for text on strong colors
                txt_color = "white" if abs(zval) > 0.6*float(vmax) else "black"
                weight = "bold" if nval>5 else "normal"
                txt = f"{text_fmt.format(zval)}\n(n={nval}"
                #if low_n is not None and nval and nval < low_n:
                #    txt += "*"
                txt += ")"
                ax.text(c, r, txt, ha="center", va="center",
                        color=txt_color, fontsize=fontsize, fontweight=weight, linespacing=0.9)

    # Title (panel letter added outside in the driver if you prefer)
    titles = {"PP": r"(a) $\bar{z}_{p,c}^{PP}$", "NP": r"(b) $\bar{z}_{p,c}^{NP}$", "U": r"(c) $\bar{z}_{p,c}^{U}$"}
    ax.set_title(titles.get(klass, f"mean z — {klass}"), fontsize=fontsize+3, pad=6)

    return im, ax, (vmin, vmax)

def compute_z_by_seq(df: pd.DataFrame, klass: str, alpha=0.05, fdr=True):
    """
    df must contain: ['seq_id','feature','dimension','m_pc','phi'] and already be filtered
    to the features of interest if desired (e.g., the 4 rows you're using).
    
    Returns:
      z_tab: 4x6 DataFrame of across-sequence mean z for the class
      n_img: 4x6 DataFrame of total image counts used
      n_seq: 4x6 DataFrame of number of sequences contributing (1..5)
      p_tab: 4x6 DataFrame of p-values for H0: mean z == 0 (per cell; based on per-sequence means)
      sig_tab: 4x6 boolean DataFrame of significance after (optional) FDR
    """
    # Per-sequence normalization for each (feature, dimension)
    stats = (df.groupby(["seq_id","feature","dimension"], observed=True)["phi"]
               .agg(mu="mean", sigma=lambda s: s.std(ddof=1))
               .reset_index())
    dfn = df.merge(stats, on=["seq_id","feature","dimension"], how="left")
    eps = 1e-9
    den = dfn["sigma"].where(dfn["sigma"].notna() & (dfn["sigma"] > 0), eps)
    dfn["z"] = (dfn["phi"] - dfn["mu"]) / den

    # keep only the class
    dfn = dfn[dfn["m_pc"] == klass].copy()

    # (1) per-sequence mean z for each cell, and image counts
    per_seq = (dfn.groupby(["feature","dimension","seq_id"], observed=True)
                  .agg(mean_z=("z","mean"), n_img=("z","size"))
                  .reset_index())

    # (2) aggregate across sequences: mean of per-seq means, n_img total, n_seq
    agg = (per_seq.groupby(["feature","dimension"], observed=True)
                  .agg(z=("mean_z","mean"),
                       n_img=("n_img","sum"),
                       n_seq=("mean_z","count"))
                  .reset_index())

    # (3) significance: t-test across sequences vs 0 (uses per-seq means)
    pvals = []
    for _, row in agg.iterrows():
        sub = per_seq[(per_seq["feature"]==row["feature"]) &
                      (per_seq["dimension"]==row["dimension"])]["mean_z"].dropna().to_numpy()
        if sub.size >= 2:  # need at least 2 sequences to test
            # scipy ttest_1samp is fine; for n=5 it's Welch-equivalent in 1-sample case
            t, p = ttest_1samp(sub, popmean=0.0, nan_policy="omit")
        else:
            p = np.nan
        pvals.append(p)
    agg["pval"] = pvals

    # (4) optional FDR per class across all cells
    if fdr:
        mask = agg["pval"].notna()
        _, q, _, _ = multipletests(agg.loc[mask,"pval"].to_numpy(), method="fdr_bh")
        agg.loc[mask,"qval"] = q
        alpha_col = "qval"
    else:
        alpha_col = "pval"

    # Build 4×6 tables
    z_tab   = (agg.pivot(index="feature", columns="dimension", values="z")
                  .reindex(index=CATS, columns=DIMS))
    n_img_t = (agg.pivot(index="feature", columns="dimension", values="n_img")
                  .reindex(index=CATS, columns=DIMS))
    n_seq_t = (agg.pivot(index="feature", columns="dimension", values="n_seq")
                  .reindex(index=CATS, columns=DIMS))
    p_tab   = (agg.pivot(index="feature", columns="dimension", values=alpha_col)
                  .reindex(index=CATS, columns=DIMS))
    sig_tab = p_tab < alpha

    return z_tab, n_img_t, n_seq_t, p_tab, sig_tab


def z_legend_handles(
    *,
    cmap_red: str = "#F14272",
    cmap_blue: str = "#3B4EB6",
    nan_color: str = "#D8D8D8",
    show_nan: bool = True,
    show_threshold_note: bool = True
):
    """
    Legend swatches for the within-sequence z-score panels.
    - red/blue = higher/lower mean z
    - bold text = passes class threshold (PP: z≥0.5; NP: z≤−0.5; U: |z|≤0.2)
    - '*' after n = low sample size (if enabled in plot)
    """
    handles = [
        Patch(facecolor=cmap_red,  edgecolor=cmap_red,  label="higher z"),
        Patch(facecolor=cmap_blue, edgecolor=cmap_blue, label="lower z"),
        Patch(facecolor="none", edgecolor="none", label="bold = passes class threshold"),
        Patch(facecolor="none", edgecolor="none", label="* = low n"),
    ]
    if show_nan:
        handles.append(Patch(facecolor=nan_color, edgecolor=nan_color, label="no data"))
    if not show_threshold_note:
        # remove the threshold note line if you don't want it
        handles = [h for h in handles if "passes class threshold" not in h.get_label()]
    return handles