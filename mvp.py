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
from litellm import completion
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


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str


@dataclass
class RunConfig:
    models: List[ModelSpec]
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

    model_payloads = raw.get("models")
    models: List[ModelSpec] = []
    if isinstance(model_payloads, list) and model_payloads:
        for item in model_payloads:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            provider = str(item.get("provider", "")).strip().lower()
            model = str(item.get("model", "")).strip()
            if name and provider and model:
                models.append(ModelSpec(name=name, provider=provider, model=model))

    # Backward compatibility with legacy single-model config.
    if not models:
        legacy_model = str(raw.get("llm_model", "gpt-4o")).strip()
        models = [ModelSpec(name="openai_default", provider="openai", model=legacy_model)]

    return RunConfig(
        models=models,
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
    if score < 1 or score > 7:
        raise ValueError("Score must be between 1 and 7")

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


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
            else:
                chunks.append(str(block))
        return "\n".join([c for c in chunks if c]).strip()
    return str(content or "")


def call_llm(
    model_spec: ModelSpec,
    prompt: str,
    timeout: int,
) -> Tuple[str, Dict[str, int]]:
    litellm_model = (
        model_spec.model
        if model_spec.model.startswith(f"{model_spec.provider}/")
        else f"{model_spec.provider}/{model_spec.model}"
    )

    response = completion(
        model=litellm_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        timeout=timeout,
        drop_params=True,
    )

    message_content = response.choices[0].message.content
    content = normalize_message_content(message_content)
    usage = getattr(response, "usage", None)
    usage_payload = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return content, usage_payload

def get_available_models(cfg: RunConfig, logger: RunLogger) -> List[ModelSpec]:
    available: List[ModelSpec] = []
    for model_spec in cfg.models:
        required_var = PROVIDER_ENV_VARS.get(model_spec.provider)
        if required_var and not os.environ.get(required_var):
            logger.add_warning(
                {
                    "type": "missing_provider_api_key",
                    "model_name": model_spec.name,
                    "provider": model_spec.provider,
                    "model": model_spec.model,
                    "required_env_var": required_var,
                    "message": f"Skipped model because {required_var} is not set.",
                }
            )
            continue
        available.append(model_spec)
    return available

def run_single_judgment(
    *,
    model_spec: ModelSpec,
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
            raw_text, usage = call_llm(model_spec, rendered, cfg.request_timeout)
            parsed = extract_json_block(raw_text)
            score, reason = parse_score_reason(parsed)
            logger.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
            return {
                "model_name": model_spec.name,
                "provider": model_spec.provider,
                "model_id": model_spec.model,
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
            "model_name": model_spec.name,
            "provider": model_spec.provider,
            "model": model_spec.model,
            "prompt_name": prompt_name,
            "product_id": product_id,
            "dimension": dimension,
            "repeat": repeat_idx,
            "error": last_error,
        }
    )
    return {
        "model_name": model_spec.name,
        "provider": model_spec.provider,
        "model_id": model_spec.model,
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
    active_models = get_available_models(cfg, logger)
    if not active_models:
        required_vars = sorted(set(PROVIDER_ENV_VARS.values()))
        raise EnvironmentError(
            "No configured models are runnable. Set at least one provider API key: "
            + ", ".join(required_vars)
        )

    tasks: List[Tuple[ModelSpec, str, str, pd.Series, str, int]] = []
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

        for model_spec in active_models:
            for _, row in products_df.iterrows():
                for dimension in cfg.dimensions:
                    for repeat_idx in range(1, cfg.judge_repeats + 1):
                        tasks.append((model_spec, prompt_name, template, row, dimension, repeat_idx))

    raw_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [
            pool.submit(
                run_single_judgment,
                model_spec=t[0],
                cfg=cfg,
                prompt_name=t[1],
                template=t[2],
                product_row=t[3],
                dimension=t[4],
                repeat_idx=t[5],
                logger=logger,
            )
            for t in tasks
        ]
        for fut in as_completed(futures):
            raw_rows.append(fut.result())

    raw_df = pd.DataFrame(raw_rows)

    valid_df = raw_df.dropna(subset=["score"]).copy()
    if valid_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "model_name",
                    "provider",
                    "model_id",
                    "prompt_name",
                    "product_id",
                    "dimension",
                    "llm_score",
                    "reason",
                ]
            ),
            raw_df,
        )

    grouped = (
        valid_df.groupby(
            ["model_name", "provider", "model_id", "prompt_name", "product_id", "dimension"],
            as_index=False,
        )
        .agg(
            llm_score=("score", "mean"),
            reason=("reason", lambda s: " | ".join([v for v in s.head(3) if v])),
        )
        .sort_values(["model_name", "prompt_name", "product_id", "dimension"])
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
