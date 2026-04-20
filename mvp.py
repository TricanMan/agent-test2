import argparse
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Tuple
from scipy.stats import f_oneway


REQUIRED_PRODUCT_COLUMNS = {"product_name", "description", "category"}

REQUIRED_HUMAN_COLUMNS = {
    "product_id",
    "rater_id",
    "physical_score",
    "cognitive_score",
    "reason",
}


PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVP prompt-evaluation pipeline for LLM-as-judge testing"
    )
    parser.add_argument("--products", required=True, help="Path to products CSV")
    parser.add_argument("--human", default=None, help="Path to human ratings CSV")
    parser.add_argument("--prompts", required=True, help="Path to prompt variants JSON")
    parser.add_argument(
        "--config", default="config.json", help="Path to config JSON (default: config.json)"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for generated artifacts"
    )
    return parser.parse_args()


def generalized_fleiss_kappa(counts: pd.DataFrame) -> float:
    if counts.empty:
        return math.nan

    mat = counts.to_numpy(dtype=float)
    row_totals = mat.sum(axis=1)
    if (row_totals < 2).any():
        return math.nan

    p_i = (mat * (mat - 1)).sum(axis=1) / (row_totals * (row_totals - 1))
    p_bar = p_i.mean()

    p_j = mat.sum(axis=0) / row_totals.sum()
    p_e = (p_j**2).sum()
    denom = 1 - p_e
    if denom == 0:
        return math.nan
    return float((p_bar - p_e) / denom)


def make_count_matrix(
    scores_df: pd.DataFrame,
    item_col: str,
    score_col: str,
    categories: List[int],
) -> pd.DataFrame:
    score_df = scores_df.copy()
    score_df[score_col] = pd.to_numeric(score_df[score_col], errors="coerce").round().astype("Int64")
    score_df = score_df[score_df[score_col].isin(categories)]
    if score_df.empty:
        return pd.DataFrame(columns=categories)

    counts = (
        score_df.groupby([item_col, score_col]).size().unstack(fill_value=0).reindex(columns=categories, fill_value=0)
    )
    return counts


def compute_metrics(
    llm_agg_df: pd.DataFrame,
    llm_raw_df: pd.DataFrame,
    human_df: pd.DataFrame,
    cfg: RunConfig,
) -> pd.DataFrame:
    # Preserved for backward-compatible function signature.
    _ = llm_raw_df
    _ = human_df

    results: List[Dict[str, Any]] = []

    for prompt_name, prompt_df in llm_agg_df.groupby("prompt_name"):
        prompt_metrics: Dict[str, Any] = {"prompt_name": prompt_name}
        p_values: List[float] = []
        f_values: List[float] = []

        for dim in cfg.dimensions:
            dim_df = prompt_df[prompt_df["dimension"] == dim][
                ["model_name", "product_id", "llm_score"]
            ].copy()
            dim_df["llm_score"] = pd.to_numeric(dim_df["llm_score"], errors="coerce")
            dim_df = dim_df.dropna(subset=["llm_score"])

            group_scores: List[List[float]] = []
            for _, model_scores in dim_df.groupby("model_name"):
                values = model_scores["llm_score"].tolist()
                if len(values) >= 2:
                    group_scores.append(values)

            prompt_metrics[f"model_groups_{dim}"] = int(len(group_scores))
            prompt_metrics[f"n_obs_{dim}"] = int(len(dim_df))

            if len(group_scores) >= 2:
                f_stat, p_val = f_oneway(*group_scores)
                f_stat = float(f_stat)
                p_val = float(p_val)
            else:
                f_stat = math.nan
                p_val = math.nan

            prompt_metrics[f"anova_f_{dim}"] = f_stat
            prompt_metrics[f"anova_p_{dim}"] = p_val

            if not math.isnan(p_val):
                p_values.append(p_val)
            if not math.isnan(f_stat):
                f_values.append(f_stat)

        prompt_metrics["min_anova_p"] = min(p_values) if p_values else math.nan
        prompt_metrics["max_anova_f"] = max(f_values) if f_values else math.nan
        prompt_metrics["significant_any_dim"] = any(p < 0.05 for p in p_values) if p_values else False

        results.append(prompt_metrics)

    metrics_df = pd.DataFrame(results)
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_name",
                "anova_f_physical",
                "anova_p_physical",
                "anova_f_cognitive",
                "anova_p_cognitive",
                "min_anova_p",
                "max_anova_f",
                "significant_any_dim",
            ]
        )

    metrics_df = metrics_df.sort_values(["min_anova_p", "max_anova_f"], ascending=[True, False])
    return metrics_df

from src.run_handler import RunConfig
def save_scatter_plots(
    llm_agg_df: pd.DataFrame,
    human_df: pd.DataFrame,
    cfg: RunConfig,
    output_dir: Path,
) -> List[str]:
    def safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    _ = human_df

    generated_files: List[str] = []
    for prompt_name, prompt_rows in llm_agg_df.groupby("prompt_name"):
        fig, axes = plt.subplots(1, len(cfg.dimensions), figsize=(6 * len(cfg.dimensions), 5))
        if len(cfg.dimensions) == 1:
            axes = [axes]

        plotted = False
        for idx, dim in enumerate(cfg.dimensions):
            dim_llm = prompt_rows[prompt_rows["dimension"] == dim][["model_name", "llm_score"]].copy()
            dim_llm["llm_score"] = pd.to_numeric(dim_llm["llm_score"], errors="coerce")
            dim_llm = dim_llm.dropna(subset=["llm_score"])

            ax = axes[idx]
            if dim_llm.empty:
                ax.set_title(f"{dim}: no overlap")
                ax.set_xlabel("Model")
                ax.set_ylabel("LLM score")
                continue

            plotted = True
            sns.boxplot(data=dim_llm, x="model_name", y="llm_score", ax=ax)
            sns.stripplot(data=dim_llm, x="model_name", y="llm_score", ax=ax, color="black", alpha=0.5)
            ax.set_title(f"{prompt_name} | {dim}")
            ax.set_xlabel("Model")
            ax.set_ylabel("LLM score")
            ax.set_ylim(1, 7)
            ax.tick_params(axis="x", rotation=20)

        if plotted:
            fig.tight_layout()
            out_path = output_dir / f"anova_groups_{safe_name(prompt_name)}.png"
            fig.savefig(out_path, dpi=150)
            generated_files.append(str(out_path))
        plt.close(fig)

    return generated_files


def write_summary_report(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    plot_paths: List[str],
) -> None:
    summary_path = output_dir / "summary_report.md"

    if metrics_df.empty:
        content = "# Summary Report\n\nNo metrics were generated. Check logs.json for errors.\n"
        summary_path.write_text(content, encoding="utf-8")
        return

    top_prompt = metrics_df.iloc[0]
    lines = [
        "# Summary Report",
        "",
        "## Strongest Between-Model Difference",
        f"- Prompt: **{top_prompt['prompt_name']}**",
        f"- Minimum ANOVA p-value across dimensions: **{top_prompt.get('min_anova_p', float('nan')):.6f}**",
        f"- Maximum ANOVA F-statistic across dimensions: **{top_prompt.get('max_anova_f', float('nan')):.3f}**",
        "",
        "## Ranking (by min ANOVA p-value)",
        "",
    ]

    show_cols = [
        c
        for c in [
            "prompt_name",
            "anova_f_physical",
            "anova_p_physical",
            "anova_f_cognitive",
            "anova_p_cognitive",
            "min_anova_p",
            "max_anova_f",
            "significant_any_dim",
        ]
        if c in metrics_df.columns
    ]
    lines.append(metrics_df[show_cols].to_markdown(index=False))

    if plot_paths:
        lines.extend(["", "## Group Comparison Plots"]) 
        for p in plot_paths:
            rel = Path(p).name
            lines.append(f"- ![{rel}]({rel})")

    significant = metrics_df[metrics_df["significant_any_dim"] == True]
    lines.extend(["", "## Significance Flags"])
    if significant.empty:
        lines.append("- No prompts showed significant between-model differences at p < 0.05.")
    else:
        for _, row in significant.iterrows():
            lines.append(
                f"- {row['prompt_name']} shows significant between-model differences (min p={row['min_anova_p']:.6f})"
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



from src.input_processing import validate_human, validate_products, ensure_product_id
from src.run_log import RunLogger
from src.run_handler import load_config, load_json, run_prompt_runner, write_logs

def main() -> None:
    # Load environment variables from .env if present.
    load_dotenv()

    args = parse_args()
    logger = RunLogger()

    cfg = load_config(args.config)

    products_df = pd.read_csv(args.products)
    human_df: Optional[pd.DataFrame] = None
    if args.human:
        human_df = pd.read_csv(args.human)
        human_df = validate_human(human_df)
    prompts = load_json(args.prompts)

    products_df = validate_products(products_df)
    products_df = ensure_product_id(products_df, logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_agg_df, llm_raw_df = run_prompt_runner(products_df, prompts, cfg, logger)

    llm_output = llm_agg_df[
        [
            "model_name",
            "provider",
            "model_id",
            "prompt_name",
            "product_id",
            "dimension",
            "llm_score",
            "reason",
        ]
    ]
    llm_output.to_csv(output_dir / "llm_ratings.csv", index=False)

    metrics_df = compute_metrics(llm_agg_df, llm_raw_df, human_df, cfg)
    metrics_df.to_csv(output_dir / "metrics_table.csv", index=False)

    plot_paths = save_scatter_plots(llm_agg_df, human_df, cfg, output_dir)
    write_summary_report(metrics_df, output_dir, plot_paths)
    write_logs(logger, cfg, output_dir)


if __name__ == "__main__":
    main()
