from litellm import completion
import pandas as pd
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time, os, json
from pathlib import Path
from src.run_log import RunLogger
from dotenv import load_dotenv

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

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


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

from .output_processing import extract_json_block, parse_score_reason

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
