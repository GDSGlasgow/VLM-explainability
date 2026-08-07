"""Summarise and plot repeated InternVL reliability measurements."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("safe", "lively", "beauty", "wealthy", "depressing", "boring")
DIMENSION_COLORS = {
    "safe": "#3B4EB6",
    "lively": "#F14272",
    "beauty": "#15CA7A",
    "wealthy": "#ECACBE",
    "depressing": "#ACB4E2",
    "boring": "#444444",
}
BLUE = "#3B4EB6"
PINK = "#F14272"
NEUTRAL = "#444444"


def load_reliability_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the wide CSV, validate it, and return wide and long representations."""
    wide = pd.read_csv(path)
    required_metadata = {"sequence_id", "image_id", "run"}
    required_values = {
        f"{dimension}_{field}"
        for dimension in DIMENSIONS
        for field in ("score", "confidence")
    }
    missing = sorted((required_metadata | required_values) - set(wide.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    duplicate = wide.duplicated(["sequence_id", "image_id", "run"])
    if duplicate.any():
        raise ValueError(f"Found {int(duplicate.sum())} duplicate image/run rows")

    records = []
    for dimension in DIMENSIONS:
        rationale_column = f"{dimension}_rationale"
        part = wide[["sequence_id", "image_id", "run"]].copy()
        part["dimension"] = dimension
        part["score"] = pd.to_numeric(wide[f"{dimension}_score"], errors="coerce")
        part["confidence"] = pd.to_numeric(
            wide[f"{dimension}_confidence"], errors="coerce"
        )
        part["rationale"] = (
            wide[rationale_column].fillna("").astype(str)
            if rationale_column in wide.columns
            else ""
        )
        records.append(part)

    long = pd.concat(records, ignore_index=True)
    long["image_key"] = (
        long["sequence_id"].astype(str) + "/" + long["image_id"].astype(str)
    )
    return wide, long


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def summarise_by_image_dimension(long: pd.DataFrame) -> pd.DataFrame:
    """Calculate within-image variability for every dimension."""
    grouped = long.groupby(
        ["sequence_id", "image_id", "image_key", "dimension"], sort=False
    )
    summary = grouped.agg(
        n_runs=("run", "nunique"),
        n_scores=("score", "count"),
        score_mean=("score", "mean"),
        score_sd=("score", "std"),
        score_variance=("score", "var"),
        score_median=("score", "median"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_iqr=("score", _iqr),
        score_unique_values=("score", "nunique"),
        n_confidences=("confidence", "count"),
        confidence_mean=("confidence", "mean"),
        confidence_sd=("confidence", "std"),
        confidence_variance=("confidence", "var"),
        confidence_median=("confidence", "median"),
        confidence_min=("confidence", "min"),
        confidence_max=("confidence", "max"),
        confidence_iqr=("confidence", _iqr),
        confidence_unique_values=("confidence", "nunique"),
    ).reset_index()
    summary["score_range"] = summary["score_max"] - summary["score_min"]
    summary["confidence_range"] = (
        summary["confidence_max"] - summary["confidence_min"]
    )
    return summary


def _pooled_within_sd(group: pd.DataFrame, value: str) -> float:
    residual_sum_squares = 0.0
    residual_df = 0
    for _, image in group.groupby("image_key"):
        values = image[value].dropna().to_numpy(dtype=float)
        if len(values) > 1:
            residual_sum_squares += float(np.square(values - values.mean()).sum())
            residual_df += len(values) - 1
    return float(np.sqrt(residual_sum_squares / residual_df)) if residual_df else np.nan


def _icc_one_way(group: pd.DataFrame, value: str) -> float:
    """One-way random-effects single-measure ICC, allowing unequal group sizes."""
    valid = group[["image_key", value]].dropna()
    image_groups = list(valid.groupby("image_key")[value])
    n_images = len(image_groups)
    n_total = len(valid)
    if n_images < 2 or n_total <= n_images:
        return np.nan

    grand_mean = float(valid[value].mean())
    between_ss = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2
        for _, values in image_groups
    )
    within_ss = sum(
        float(np.square(values.to_numpy(dtype=float) - float(values.mean())).sum())
        for _, values in image_groups
    )
    ms_between = between_ss / (n_images - 1)
    ms_within = within_ss / (n_total - n_images)
    sizes = np.array([len(values) for _, values in image_groups], dtype=float)
    effective_k = (n_total - np.square(sizes).sum() / n_total) / (n_images - 1)
    denominator = ms_between + (effective_k - 1) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator else np.nan


def summarise_by_dimension(
    long: pd.DataFrame, image_summary: pd.DataFrame
) -> pd.DataFrame:
    """Summarise variability and repeatability across images for each dimension."""
    rows = []
    for dimension in DIMENSIONS:
        raw = long[long["dimension"] == dimension]
        aggregate = image_summary[image_summary["dimension"] == dimension]
        score_within_sd = _pooled_within_sd(raw, "score")
        confidence_within_sd = _pooled_within_sd(raw, "confidence")
        rows.append(
            {
                "dimension": dimension,
                "n_images": aggregate["image_key"].nunique(),
                "min_attempted_runs_per_image": aggregate["n_runs"].min(),
                "max_attempted_runs_per_image": aggregate["n_runs"].max(),
                "min_valid_scores_per_image": aggregate["n_scores"].min(),
                "max_valid_scores_per_image": aggregate["n_scores"].max(),
                "mean_score": raw["score"].mean(),
                "pooled_within_image_score_sd": score_within_sd,
                "mean_image_score_sd": aggregate["score_sd"].mean(),
                "median_image_score_sd": aggregate["score_sd"].median(),
                "max_image_score_sd": aggregate["score_sd"].max(),
                "mean_image_score_range": aggregate["score_range"].mean(),
                "score_repeatability_coefficient_95": 1.96
                * np.sqrt(2)
                * score_within_sd,
                "score_icc_1_1": _icc_one_way(raw, "score"),
                "mean_confidence": raw["confidence"].mean(),
                "pooled_within_image_confidence_sd": confidence_within_sd,
                "mean_image_confidence_sd": aggregate["confidence_sd"].mean(),
                "median_image_confidence_sd": aggregate["confidence_sd"].median(),
                "max_image_confidence_sd": aggregate["confidence_sd"].max(),
                "mean_image_confidence_range": aggregate["confidence_range"].mean(),
                "confidence_icc_1_1": _icc_one_way(raw, "confidence"),
            }
        )
    return pd.DataFrame(rows)


def _correlations(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    result: dict[str, float | int] = {"n": len(valid)}
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return {
            **result,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = pearsonr(valid["x"], valid["y"])
        spearman = spearmanr(valid["x"], valid["y"])
    return {
        **result,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def confidence_variability_correlations(
    long: pd.DataFrame, image_summary: pd.DataFrame
) -> pd.DataFrame:
    """Test whether greater stated confidence corresponds to more stable scores."""
    rows = []
    scopes = [("all_dimensions", "all", image_summary)] + [
        ("single_dimension", dimension, image_summary[image_summary["dimension"] == dimension])
        for dimension in DIMENSIONS
    ]
    for scope, dimension, data in scopes:
        rows.append(
            {
                "analysis_level": "image_dimension",
                "scope": scope,
                "dimension": dimension,
                "x": "mean_confidence",
                "y": "score_sd",
                "hypothesized_direction": "negative",
                **_correlations(data["confidence_mean"], data["score_sd"]),
            }
        )

    # The pooled association can be partly driven by systematic differences
    # between dimensions. Center both variables within dimension as a simple
    # dimension-adjusted sensitivity analysis.
    adjusted = image_summary.copy()
    adjusted["confidence_mean_centered_by_dimension"] = (
        adjusted["confidence_mean"]
        - adjusted.groupby("dimension")["confidence_mean"].transform("mean")
    )
    adjusted["score_sd_centered_by_dimension"] = (
        adjusted["score_sd"]
        - adjusted.groupby("dimension")["score_sd"].transform("mean")
    )
    rows.append(
        {
            "analysis_level": "image_dimension_adjusted",
            "scope": "all_dimensions",
            "dimension": "all",
            "x": "mean_confidence_centered_by_dimension",
            "y": "score_sd_centered_by_dimension",
            "hypothesized_direction": "negative",
            **_correlations(
                adjusted["confidence_mean_centered_by_dimension"],
                adjusted["score_sd_centered_by_dimension"],
            ),
        }
    )

    # Secondary diagnostic: within each image/dimension, do lower-confidence runs
    # deviate further from that image/dimension's mean score?
    run_level = long.copy()
    group_keys = ["sequence_id", "image_id", "dimension"]
    run_level["score_abs_deviation"] = (
        run_level["score"]
        - run_level.groupby(group_keys)["score"].transform("mean")
    ).abs()
    run_level["confidence_centered"] = (
        run_level["confidence"]
        - run_level.groupby(group_keys)["confidence"].transform("mean")
    )
    run_scopes = [("all_dimensions", "all", run_level)] + [
        ("single_dimension", dimension, run_level[run_level["dimension"] == dimension])
        for dimension in DIMENSIONS
    ]
    for scope, dimension, data in run_scopes:
        rows.append(
            {
                "analysis_level": "run_within_image_dimension",
                "scope": scope,
                "dimension": dimension,
                "x": "confidence_centered",
                "y": "absolute_score_deviation",
                "hypothesized_direction": "negative",
                **_correlations(data["confidence_centered"], data["score_abs_deviation"]),
            }
        )

    result = pd.DataFrame(rows)
    for p_column in ("pearson_p", "spearman_p"):
        q_column = p_column.replace("_p", "_q_bh")
        result[q_column] = np.nan
        for analysis_level, indices in result.groupby("analysis_level").groups.items():
            valid_indices = [index for index in indices if pd.notna(result.loc[index, p_column])]
            if valid_indices:
                result.loc[valid_indices, q_column] = multipletests(
                    result.loc[valid_indices, p_column], method="fdr_bh"
                )[1]
    return result


def _similarity_by_group(
    matrix, metadata: pd.DataFrame, metric_prefix: str
) -> pd.DataFrame:
    """Compare within-image similarity with a same-dimension/different-image baseline."""
    rows = []
    for dimension in DIMENSIONS:
        dimension_positions = np.flatnonzero(
            metadata["dimension"].to_numpy() == dimension
        )
        dimension_images = metadata.iloc[dimension_positions]["image_key"].to_numpy()
        for image_key in pd.unique(dimension_images):
            local_mask = dimension_images == image_key
            within_positions = dimension_positions[local_mask]
            baseline_positions = dimension_positions[~local_mask]
            if len(within_positions) < 2 or len(baseline_positions) == 0:
                continue

            within_matrix = cosine_similarity(matrix[within_positions])
            triangle = within_matrix[np.triu_indices_from(within_matrix, k=1)]
            between = cosine_similarity(
                matrix[within_positions], matrix[baseline_positions]
            ).ravel()
            sequence_id, image_id = image_key.split("/", maxsplit=1)
            within_mean = float(np.mean(triangle))
            baseline_mean = float(np.mean(between))
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "image_id": image_id,
                    "image_key": image_key,
                    "dimension": dimension,
                    "n_rationales": len(within_positions),
                    f"{metric_prefix}_within_mean_cosine": within_mean,
                    f"{metric_prefix}_within_median_cosine": float(np.median(triangle)),
                    f"{metric_prefix}_between_image_baseline_mean_cosine": baseline_mean,
                    f"{metric_prefix}_specificity_delta": within_mean - baseline_mean,
                    f"{metric_prefix}_within_cosine_distance": 1.0 - within_mean,
                }
            )
    return pd.DataFrame(rows)


def rationale_similarity(
    long: pd.DataFrame, semantic_model: str | None = None, batch_size: int = 64
) -> pd.DataFrame:
    """Measure lexical and optionally semantic consistency of model rationales."""
    rationales = long[long["rationale"].str.strip().ne("")].reset_index(drop=True)
    if rationales.empty:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(rationales["rationale"])
    result = _similarity_by_group(tfidf, rationales, "tfidf")

    if semantic_model:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic analysis requires sentence-transformers. "
                "Install the updated requirements.txt or omit --semantic-model."
            ) from exc
        model = SentenceTransformer(semantic_model)
        embeddings = model.encode(
            rationales["rationale"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        semantic = _similarity_by_group(embeddings, rationales, "semantic")
        result = result.merge(
            semantic,
            on=["sequence_id", "image_id", "image_key", "dimension", "n_rationales"],
            how="outer",
        )
    return result


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)


def plot_variability(image_summary: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    rng = np.random.default_rng(42)
    metrics = [
        ("score_sd", "Within-image score SD"),
        ("confidence_sd", "Within-image confidence SD"),
    ]
    positions = np.arange(len(DIMENSIONS))
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = [
            image_summary.loc[image_summary["dimension"] == dimension, metric]
            .dropna()
            .to_numpy()
            for dimension in DIMENSIONS
        ]
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.8},
            whiskerprops={"color": NEUTRAL},
            capprops={"color": NEUTRAL},
        )
        for box, dimension in zip(boxes["boxes"], DIMENSIONS):
            box.set_facecolor(DIMENSION_COLORS[dimension])
            box.set_alpha(0.82)
        for position, dimension, dimension_values in zip(positions, DIMENSIONS, values):
            jitter = rng.uniform(-0.12, 0.12, len(dimension_values))
            ax.scatter(
                position + jitter,
                dimension_values,
                s=24,
                facecolor="white",
                edgecolor=DIMENSION_COLORS[dimension],
                linewidth=0.9,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(DIMENSIONS, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        _clean_axes(ax)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_variability_heatmaps(image_summary: pd.DataFrame, output_stem: Path) -> None:
    ordered_images = image_summary[["sequence_id", "image_id", "image_key"]].drop_duplicates()
    labels = [
        f"{str(row.sequence_id)[:6]}…/{row.image_id}"
        for row in ordered_images.itertuples(index=False)
    ]
    cmap = LinearSegmentedColormap.from_list(
        "project_blue_pink", [BLUE, "#FFFFFF", PINK], N=256
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), dpi=160)
    for ax, metric, title in zip(
        axes,
        ("score_sd", "confidence_sd"),
        ("Score SD", "Confidence SD"),
    ):
        matrix = (
            image_summary.pivot(index="image_key", columns="dimension", values=metric)
            .reindex(index=ordered_images["image_key"], columns=DIMENSIONS)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xticks(np.arange(len(DIMENSIONS)))
        ax.set_xticklabels(DIMENSIONS, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xticks(np.arange(-0.5, len(DIMENSIONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_confidence_relationship(
    image_summary: pd.DataFrame, correlations: pd.DataFrame, output_stem: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    for dimension in DIMENSIONS:
        data = image_summary[image_summary["dimension"] == dimension]
        ax.scatter(
            data["confidence_mean"],
            data["score_sd"],
            s=46,
            color=DIMENSION_COLORS[dimension],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=dimension,
        )
    valid = image_summary[["confidence_mean", "score_sd"]].dropna()
    if len(valid) > 1 and valid["confidence_mean"].nunique() > 1:
        slope, intercept = np.polyfit(valid["confidence_mean"], valid["score_sd"], 1)
        x_line = np.linspace(valid["confidence_mean"].min(), valid["confidence_mean"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=NEUTRAL, linestyle="--", linewidth=1.3)

    overall = correlations[
        (correlations["analysis_level"] == "image_dimension")
        & (correlations["dimension"] == "all")
    ].iloc[0]
    adjusted = correlations[
        (correlations["analysis_level"] == "image_dimension_adjusted")
        & (correlations["dimension"] == "all")
    ].iloc[0]
    ax.text(
        0.03,
        0.97,
        "Pooled: "
        f"ρ = {overall['spearman_rho']:.2f}, p = {overall['spearman_p']:.3g}\n"
        "Dimension-adjusted: "
        f"ρ = {adjusted['spearman_rho']:.2f}, p = {adjusted['spearman_p']:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "alpha": 0.9},
    )
    ax.set_xlabel("Mean VLM confidence")
    ax.set_ylabel("Within-image score SD")
    ax.legend(frameon=False, ncol=2, title="Dimension")
    _clean_axes(ax)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_rationale_similarity(similarity: pd.DataFrame, output_stem: Path) -> None:
    metric_prefix = "semantic" if "semantic_within_mean_cosine" in similarity else "tfidf"
    within_column = f"{metric_prefix}_within_mean_cosine"
    baseline_column = f"{metric_prefix}_between_image_baseline_mean_cosine"
    means = similarity.groupby("dimension")[[within_column, baseline_column]].mean().reindex(DIMENSIONS)
    x = np.arange(len(DIMENSIONS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.3), dpi=160)
    ax.bar(x - width / 2, means[within_column], width, color=PINK, label="Same image")
    ax.bar(
        x + width / 2,
        means[baseline_column],
        width,
        color=BLUE,
        label="Different-image baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSIONS, rotation=35, ha="right")
    ax.set_ylabel(f"Mean {metric_prefix.upper()} cosine similarity")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _save_figure(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def data_quality_summary(wide: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    def nonempty(column: str) -> int:
        if column not in wide.columns:
            return 0
        return int(wide[column].fillna("").astype(str).str.strip().ne("").sum())

    score_confidence_columns = [
        f"{dimension}_{field}"
        for dimension in DIMENSIONS
        for field in ("score", "confidence")
    ]
    complete_rows = wide[score_confidence_columns].notna().all(axis=1)
    generation_errors = nonempty("generation_error")
    return pd.DataFrame(
        [
            {
                "csv_rows": len(wide),
                "images": long["image_key"].nunique(),
                "dimensions": long["dimension"].nunique(),
                "long_observations": len(long),
                "complete_successful_calls": int(complete_rows.sum()),
                "incomplete_or_failed_calls": int((~complete_rows).sum()),
                "complete_call_fraction": float(complete_rows.mean()),
                "missing_scores": int(long["score"].isna().sum()),
                "missing_confidences": int(long["confidence"].isna().sum()),
                "missing_rationales": int(long["rationale"].str.strip().eq("").sum()),
                "generation_error_rows": generation_errors,
                "generation_error_fraction": generation_errors / len(wide),
                "parse_error_rows": nonempty("parse_error"),
            }
        ]
    )


def run_analysis(
    input_path: Path,
    output_dir: Path,
    plot_dir: Path,
    semantic_model: str | None = None,
    semantic_batch_size: int = 64,
    skip_rationales: bool = False,
) -> None:
    wide, long = load_reliability_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    quality = data_quality_summary(wide, long)
    image_summary = summarise_by_image_dimension(long)
    dimension_summary = summarise_by_dimension(long, image_summary)
    correlations = confidence_variability_correlations(long, image_summary)

    quality.to_csv(output_dir / "reliability_data_quality.csv", index=False)
    if "generation_error" in wide.columns:
        error_counts = (
            wide.loc[
                wide["generation_error"].fillna("").astype(str).str.strip().ne(""),
                "generation_error",
            ]
            .value_counts()
            .rename_axis("generation_error")
            .reset_index(name="count")
        )
        error_counts.to_csv(output_dir / "reliability_generation_errors.csv", index=False)
    image_summary.to_csv(output_dir / "reliability_by_image_dimension.csv", index=False)
    dimension_summary.to_csv(output_dir / "reliability_by_dimension.csv", index=False)
    correlations.to_csv(output_dir / "reliability_confidence_correlations.csv", index=False)

    plot_variability(image_summary, plot_dir / "reliability_variability_by_dimension")
    plot_variability_heatmaps(image_summary, plot_dir / "reliability_variability_heatmaps")
    plot_confidence_relationship(
        image_summary,
        correlations,
        plot_dir / "reliability_confidence_vs_score_variability",
    )

    if not skip_rationales:
        similarity = rationale_similarity(
            long, semantic_model=semantic_model, batch_size=semantic_batch_size
        )
        if not similarity.empty:
            similarity.to_csv(
                output_dir / "reliability_rationale_similarity.csv", index=False
            )
            plot_rationale_similarity(
                similarity, plot_dir / "reliability_rationale_similarity"
            )

    overall = correlations[
        (correlations["analysis_level"] == "image_dimension")
        & (correlations["dimension"] == "all")
    ].iloc[0]
    adjusted = correlations[
        (correlations["analysis_level"] == "image_dimension_adjusted")
        & (correlations["dimension"] == "all")
    ].iloc[0]
    print(f"Analysed {len(wide)} runs across {long['image_key'].nunique()} images.")
    complete_calls = int(quality.loc[0, "complete_successful_calls"])
    if complete_calls < len(wide):
        print(
            f"WARNING: only {complete_calls}/{len(wide)} calls contain complete results; "
            "see reliability_data_quality.csv."
        )
    print(
        "Pooled mean-confidence versus score-SD: "
        f"Spearman rho={overall['spearman_rho']:.3f}, p={overall['spearman_p']:.4g}."
    )
    print(
        "Dimension-adjusted association: "
        f"Spearman rho={adjusted['spearman_rho']:.3f}, p={adjusted['spearman_p']:.4g}."
    )
    print(f"Tables: {output_dir}")
    print(f"Plots:  {plot_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse repeated InternVL scores, confidence, and rationales."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "reliability_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reliability",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "plots" / "reliability",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help=(
            "Optional sentence-transformers model, for example "
            "sentence-transformers/all-MiniLM-L6-v2. TF-IDF analysis always runs."
        ),
    )
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-rationales",
        action="store_true",
        help="Skip both lexical and optional semantic rationale analysis.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        semantic_model=args.semantic_model,
        semantic_batch_size=args.semantic_batch_size,
        skip_rationales=args.skip_rationales,
    )


if __name__ == "__main__":
    main()
