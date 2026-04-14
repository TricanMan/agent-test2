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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr


REQUIRED_PRODUCT_COLUMNS = {"product_name", "description", "category"}
REQUIRED_HUMAN_COLUMNS = {
    "product_id",
    "rater_id",
    "physical_score",
    "cognitive_score",
    "reason",
}


@dataclass
class RunConfig:
    llm_model: str
    judge_repeats: int
    dimensions: List[str]
    max_workers: int
    request_timeout: int
    retry_attempts: int
    retry_backoff_seconds: float
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


class RunLogger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_error(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.errors.append(payload)

    def add_warning(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.warnings.append(payload)

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.total_input_tokens += int(input_tokens)
            self.total_output_tokens += int(output_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVP prompt-evaluation pipeline for LLM-as-judge testing"
    )
    parser.add_argument("--products", required=True, help="Path to products CSV")
    parser.add_argument("--human", required=True, help="Path to human ratings CSV")
    parser.add_argument("--prompts", required=True, help="Path to prompt variants JSON")
    parser.add_argument(
        "--config", default="config.json", help="Path to config JSON (default: config.json)"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for generated artifacts"
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> RunConfig:
    raw = load_json(path)
    return RunConfig(
        llm_model=raw.get("llm_model", "gpt-4o"),
        judge_repeats=int(raw.get("judge_repeats", 3)),
        dimensions=list(raw.get("dimensions", ["physical", "cognitive"])),
        max_workers=int(raw.get("max_workers", 8)),
        request_timeout=int(raw.get("request_timeout", 60)),
        retry_attempts=int(raw.get("retry_attempts", 3)),
        retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 2)),
        input_cost_per_1m_tokens=float(raw.get("input_cost_per_1m_tokens", 5.0)),
        output_cost_per_1m_tokens=float(raw.get("output_cost_per_1m_tokens", 15.0)),
    )


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PRODUCT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Products CSV missing required columns: {sorted(missing)}")
    return df


def validate_human(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_HUMAN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Human ratings CSV missing required columns: {sorted(missing)}")
    return df


def ensure_product_id(products_df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    if "product_id" not in products_df.columns:
        products_df = products_df.copy()
        products_df["product_id"] = range(1, len(products_df) + 1)
        logger.add_warning(
            {
                "type": "missing_product_id",
                "message": "Products CSV did not include product_id; auto-assigned sequential ids starting at 1.",
            }
        )
    return products_df


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def parse_score_reason(payload: Dict[str, Any]) -> Tuple[int, str]:
    if "score" not in payload:
        raise ValueError("Model JSON missing 'score'")
    score = payload["score"]
    if isinstance(score, str) and score.isdigit():
        score = int(score)
    if not isinstance(score, int):
        raise ValueError("Score is not an integer")
    if score < 1 or score > 5:
        raise ValueError("Score must be between 1 and 5")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return score, reason.strip()


def build_prompt(template: str, row: pd.Series, dimension: str) -> str:
    return template.format(
        product_name=row["product_name"],
        product_desc=row["description"],
        category=row["category"],
        product_id=row["product_id"],
        dimension=dimension,
    )


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    timeout: int,
) -> Tuple[str, Dict[str, int]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        timeout=timeout,
    )

    content = response.choices[0].message.content or ""
    usage = response.usage
    usage_payload = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return content, usage_payload


def run_single_judgment(
    *,
    client: OpenAI,
    cfg: RunConfig,
    prompt_name: str,
    template: str,
    product_row: pd.Series,
    dimension: str,
    repeat_idx: int,
    logger: RunLogger,
) -> Dict[str, Any]:
    rendered = build_prompt(template, product_row, dimension)
    product_id = int(product_row["product_id"])

    last_error = None
    for attempt in range(1, cfg.retry_attempts + 1):
        try:
            raw_text, usage = call_llm(client, cfg.llm_model, rendered, cfg.request_timeout)
            parsed = extract_json_block(raw_text)
            score, reason = parse_score_reason(parsed)
            logger.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
            return {
                "prompt_name": prompt_name,
                "product_id": product_id,
                "dimension": dimension,
                "repeat": repeat_idx,
                "score": score,
                "reason": reason,
                "raw_response": raw_text,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < cfg.retry_attempts:
                time.sleep(cfg.retry_backoff_seconds * attempt)

    logger.add_error(
        {
            "type": "llm_call_failed",
            "prompt_name": prompt_name,
            "product_id": product_id,
            "dimension": dimension,
            "repeat": repeat_idx,
            "error": last_error,
        }
    )
    return {
        "prompt_name": prompt_name,
        "product_id": product_id,
        "dimension": dimension,
        "repeat": repeat_idx,
        "score": None,
        "reason": "",
        "raw_response": "",
        "error": last_error,
    }


def run_prompt_runner(
    products_df: pd.DataFrame,
    prompts: List[Dict[str, Any]],
    cfg: RunConfig,
    logger: RunLogger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    tasks: List[Tuple[str, str, pd.Series, str, int]] = []
    for prompt in prompts:
        prompt_name = prompt.get("name")
        template = prompt.get("template")
        if not prompt_name or not template:
            logger.add_warning(
                {
                    "type": "invalid_prompt_variant",
                    "payload": prompt,
                    "message": "Prompt variant skipped because it is missing name/template",
                }
            )
            continue

        for _, row in products_df.iterrows():
            for dimension in cfg.dimensions:
                for repeat_idx in range(1, cfg.judge_repeats + 1):
                    tasks.append((prompt_name, template, row, dimension, repeat_idx))

    raw_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [
            pool.submit(
                run_single_judgment,
                client=client,
                cfg=cfg,
                prompt_name=t[0],
                template=t[1],
                product_row=t[2],
                dimension=t[3],
                repeat_idx=t[4],
                logger=logger,
            )
            for t in tasks
        ]
        for fut in as_completed(futures):
            raw_rows.append(fut.result())

    raw_df = pd.DataFrame(raw_rows)

    valid_df = raw_df.dropna(subset=["score"]).copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["prompt_name", "product_id", "dimension", "llm_score", "reason"]), raw_df

    grouped = (
        valid_df.groupby(["prompt_name", "product_id", "dimension"], as_index=False)
        .agg(
            llm_score=("score", "mean"),
            reason=("reason", lambda s: " | ".join([v for v in s.head(3) if v])),
        )
        .sort_values(["prompt_name", "product_id", "dimension"])
    )

    return grouped, raw_df


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
    human_long = pd.concat(
        [
            human_df[["product_id", "rater_id", "physical_score"]]
            .rename(columns={"physical_score": "score"})
            .assign(dimension="physical"),
            human_df[["product_id", "rater_id", "cognitive_score"]]
            .rename(columns={"cognitive_score": "score"})
            .assign(dimension="cognitive"),
        ],
        ignore_index=True,
    )

    human_long["score"] = pd.to_numeric(human_long["score"], errors="coerce")
    human_long = human_long.dropna(subset=["score"])

    human_avg = (
        human_long.groupby(["product_id", "dimension"], as_index=False)
        .agg(human_avg_score=("score", "mean"))
        .sort_values(["product_id", "dimension"])
    )

    # Human-only kappa per dimension.
    categories = [1, 2, 3, 4, 5]
    human_kappa: Dict[str, float] = {}
    for dim in cfg.dimensions:
        dim_human = human_long[human_long["dimension"] == dim][["product_id", "score"]]
        counts = make_count_matrix(dim_human, "product_id", "score", categories)
        human_kappa[dim] = generalized_fleiss_kappa(counts)

    results: List[Dict[str, Any]] = []

    for prompt_name, prompt_llm in llm_agg_df.groupby("prompt_name"):
        prompt_metrics: Dict[str, Any] = {"prompt_name": prompt_name}
        corr_values = []

        for dim in cfg.dimensions:
            dim_llm = prompt_llm[prompt_llm["dimension"] == dim][
                ["product_id", "llm_score", "reason"]
            ]
            dim_human_avg = human_avg[human_avg["dimension"] == dim][
                ["product_id", "human_avg_score"]
            ]
            merged = dim_llm.merge(dim_human_avg, on="product_id", how="inner")

            if len(merged) >= 2:
                pearson_val = float(pearsonr(merged["llm_score"], merged["human_avg_score"])[0])
                spearman_val = float(spearmanr(merged["llm_score"], merged["human_avg_score"])[0])
            else:
                pearson_val = math.nan
                spearman_val = math.nan

            accuracy = (
                ((merged["llm_score"] - merged["human_avg_score"]).abs() <= 1).mean()
                if not merged.empty
                else math.nan
            )

            prompt_metrics[f"corr_{dim}"] = pearson_val
            prompt_metrics[f"spearman_{dim}"] = spearman_val
            prompt_metrics[f"accuracy_{dim}"] = float(accuracy) if not math.isnan(accuracy) else math.nan
            prompt_metrics[f"kappa_human_{dim}"] = human_kappa.get(dim, math.nan)

            if not math.isnan(pearson_val):
                corr_values.append(pearson_val)

            # Build human + LLM-raters matrix for llm-human kappa.
            llm_dim_raw = llm_raw_df[
                (llm_raw_df["prompt_name"] == prompt_name)
                & (llm_raw_df["dimension"] == dim)
                & (llm_raw_df["score"].notna())
            ][["product_id", "score"]].copy()
            llm_dim_raw["score"] = pd.to_numeric(llm_dim_raw["score"], errors="coerce")

            human_dim = human_long[human_long["dimension"] == dim][["product_id", "score"]]
            combined = pd.concat([human_dim, llm_dim_raw], ignore_index=True)
            llm_human_counts = make_count_matrix(combined, "product_id", "score", categories)
            prompt_metrics[f"kappa_llm_human_{dim}"] = generalized_fleiss_kappa(llm_human_counts)

        prompt_metrics["avg_correlation"] = (
            float(sum(corr_values) / len(corr_values)) if corr_values else math.nan
        )

        llm_human_kappas = [
            prompt_metrics.get(f"kappa_llm_human_{dim}", math.nan) for dim in cfg.dimensions
        ]
        llm_human_kappas = [v for v in llm_human_kappas if not math.isnan(v)]
        prompt_metrics["kappa"] = (
            float(sum(llm_human_kappas) / len(llm_human_kappas)) if llm_human_kappas else math.nan
        )

        prompt_metrics["low_agreement_flag"] = (
            prompt_metrics["avg_correlation"] < 0.4
            if not math.isnan(prompt_metrics["avg_correlation"])
            else True
        )

        results.append(prompt_metrics)

    metrics_df = pd.DataFrame(results)
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_name",
                "corr_physical",
                "corr_cognitive",
                "kappa",
                "avg_correlation",
                "low_agreement_flag",
            ]
        )

    metrics_df = metrics_df.sort_values("avg_correlation", ascending=False)
    return metrics_df


def save_scatter_plots(
    llm_agg_df: pd.DataFrame,
    human_df: pd.DataFrame,
    cfg: RunConfig,
    output_dir: Path,
) -> List[str]:
    human_long = pd.concat(
        [
            human_df[["product_id", "physical_score"]]
            .rename(columns={"physical_score": "score"})
            .assign(dimension="physical"),
            human_df[["product_id", "cognitive_score"]]
            .rename(columns={"cognitive_score": "score"})
            .assign(dimension="cognitive"),
        ],
        ignore_index=True,
    )
    human_long["score"] = pd.to_numeric(human_long["score"], errors="coerce")
    human_avg = human_long.groupby(["product_id", "dimension"], as_index=False)["score"].mean()

    generated_files: List[str] = []
    for prompt_name, prompt_rows in llm_agg_df.groupby("prompt_name"):
        fig, axes = plt.subplots(1, len(cfg.dimensions), figsize=(6 * len(cfg.dimensions), 5))
        if len(cfg.dimensions) == 1:
            axes = [axes]

        plotted = False
        for idx, dim in enumerate(cfg.dimensions):
            dim_llm = prompt_rows[prompt_rows["dimension"] == dim][["product_id", "llm_score"]]
            dim_human = human_avg[human_avg["dimension"] == dim][["product_id", "score"]]
            merged = dim_llm.merge(dim_human, on="product_id", how="inner")

            ax = axes[idx]
            if merged.empty:
                ax.set_title(f"{dim}: no overlap")
                ax.set_xlabel("Human avg score")
                ax.set_ylabel("LLM score")
                continue

            plotted = True
            sns.regplot(data=merged, x="score", y="llm_score", ax=ax, scatter_kws={"s": 50})
            ax.set_title(f"{prompt_name} - {dim}")
            ax.set_xlabel("Human avg score")
            ax.set_ylabel("LLM score")
            ax.set_xlim(1, 5)
            ax.set_ylim(1, 5)

        if plotted:
            fig.tight_layout()
            out_path = output_dir / f"scatter_{prompt_name}.png"
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
        "## Top Prompt Recommendation",
        f"- Name: **{top_prompt['prompt_name']}**",
        f"- Avg correlation: **{top_prompt.get('avg_correlation', float('nan')):.3f}**",
        f"- Kappa (LLM+human): **{top_prompt.get('kappa', float('nan')):.3f}**",
        "",
        "## Prompt Ranking (by avg correlation)",
        "",
    ]

    show_cols = [c for c in ["prompt_name", "corr_physical", "corr_cognitive", "kappa", "avg_correlation", "low_agreement_flag"] if c in metrics_df.columns]
    lines.append(metrics_df[show_cols].to_markdown(index=False))

    if plot_paths:
        lines.extend(["", "## Scatter Plots"]) 
        for p in plot_paths:
            rel = Path(p).name
            lines.append(f"- ![{rel}]({rel})")

    low_agreement = metrics_df[metrics_df["low_agreement_flag"] == True]
    lines.extend(["", "## Validation Flags"])
    if low_agreement.empty:
        lines.append("- No prompts fell below the 0.4 average-correlation threshold.")
    else:
        for _, row in low_agreement.iterrows():
            lines.append(
                f"- {row['prompt_name']} flagged (avg correlation={row['avg_correlation']:.3f})"
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_logs(logger: RunLogger, cfg: RunConfig, output_dir: Path) -> None:
    total_cost = (
        (logger.total_input_tokens / 1_000_000.0) * cfg.input_cost_per_1m_tokens
        + (logger.total_output_tokens / 1_000_000.0) * cfg.output_cost_per_1m_tokens
    )

    payload = {
        "warnings": logger.warnings,
        "errors": logger.errors,
        "usage": {
            "input_tokens": logger.total_input_tokens,
            "output_tokens": logger.total_output_tokens,
            "estimated_cost_usd": round(total_cost, 6),
            "pricing": {
                "input_cost_per_1m_tokens": cfg.input_cost_per_1m_tokens,
                "output_cost_per_1m_tokens": cfg.output_cost_per_1m_tokens,
            },
        },
    }

    (output_dir / "logs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    # Load environment variables from .env if present.
    load_dotenv()

    args = parse_args()
    logger = RunLogger()

    cfg = load_config(args.config)

    products_df = pd.read_csv(args.products)
    human_df = pd.read_csv(args.human)
    prompts = load_json(args.prompts)

    products_df = validate_products(products_df)
    human_df = validate_human(human_df)
    products_df = ensure_product_id(products_df, logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_agg_df, llm_raw_df = run_prompt_runner(products_df, prompts, cfg, logger)

    llm_output = llm_agg_df[["prompt_name", "product_id", "dimension", "llm_score", "reason"]]
    llm_output.to_csv(output_dir / "llm_ratings.csv", index=False)

    metrics_df = compute_metrics(llm_agg_df, llm_raw_df, human_df, cfg)
    metrics_df.to_csv(output_dir / "metrics_table.csv", index=False)

    plot_paths = save_scatter_plots(llm_agg_df, human_df, cfg, output_dir)
    write_summary_report(metrics_df, output_dir, plot_paths)
    write_logs(logger, cfg, output_dir)


if __name__ == "__main__":
    main()
