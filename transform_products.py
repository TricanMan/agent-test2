import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from groq import Groq
from ollama import Client as OllamaClient


REQUIRED_INPUT_COLUMNS = {"name", "product_url", "manual_url"}
OUTPUT_COLUMNS = ["product_name", "description", "category"]
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class TransformConfig:
    request_timeout: int = 15
    request_retries: int = 1
    retry_backoff_seconds: float = 1.5
    llm_timeout: int = 60
    max_input_chars: int = 8000
    target_words_min: int = 50
    target_words_max: int = 200
    sleep_seconds: float = 1.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    groq_model: str = "llama-3.1-8b-instant"


class RunLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.summary = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "fallback_to_groq": 0,
            "scrape_failures": 0,
        }

    def add_row(self, payload: Dict[str, Any]) -> None:
        self.rows.append(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform raw product URLs CSV into MVP-compatible products CSV"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--config", default="transform_config.json", help="Optional config JSON")
    parser.add_argument("--timeout", type=int, default=None, help="Override request timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline but do not write output")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument(
        "--log-path",
        default="outputs/transform_log.json",
        help="Path to write run diagnostics JSON",
    )
    return parser.parse_args()


def load_config(config_path: str, timeout_override: Optional[int]) -> TransformConfig:
    cfg = TransformConfig()
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg.request_timeout = int(raw.get("request_timeout", cfg.request_timeout))
        cfg.request_retries = int(raw.get("request_retries", cfg.request_retries))
        cfg.retry_backoff_seconds = float(
            raw.get("retry_backoff_seconds", cfg.retry_backoff_seconds)
        )
        cfg.llm_timeout = int(raw.get("llm_timeout", cfg.llm_timeout))
        cfg.max_input_chars = int(raw.get("max_input_chars", cfg.max_input_chars))
        cfg.target_words_min = int(raw.get("target_words_min", cfg.target_words_min))
        cfg.target_words_max = int(raw.get("target_words_max", cfg.target_words_max))
        cfg.sleep_seconds = float(raw.get("sleep_seconds", cfg.sleep_seconds))
        cfg.ollama_base_url = str(raw.get("ollama_base_url", cfg.ollama_base_url))
        cfg.ollama_model = str(raw.get("ollama_model", cfg.ollama_model))
        cfg.groq_model = str(raw.get("groq_model", cfg.groq_model))

    if timeout_override is not None:
        cfg.request_timeout = int(timeout_override)

    return cfg


def validate_input_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")


def fetch_url_text(url: str, cfg: TransformConfig) -> str:
    if not url or not isinstance(url, str):
        return ""

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    last_error = None

    for attempt in range(cfg.request_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=cfg.request_timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            # Keep this MVP focused on HTML/text pages.
            if "application/pdf" in content_type:
                return ""

            soup = BeautifulSoup(response.text, "lxml")
            selector_text = []

            title = soup.select_one("h1.product-title, h1.product-name, h1.title, h1")
            if title:
                selector_text.append(title.get_text(" ", strip=True))

            for node in soup.select(
                ".product-description, .description, .product-details, .details, [itemprop='description'], p"
            )[:15]:
                text = node.get_text(" ", strip=True)
                if text:
                    selector_text.append(text)

            if selector_text:
                joined = " ".join(selector_text)
            else:
                joined = " ".join(soup.stripped_strings)

            return re.sub(r"\s+", " ", joined).strip()
        except Exception as exc:
            last_error = str(exc)
            if attempt < cfg.request_retries:
                time.sleep(cfg.retry_backoff_seconds * (attempt + 1))

    raise RuntimeError(last_error or "Unknown fetch error")


def build_prompt(name: str, scraped_text: str, manual_text: str, cfg: TransformConfig) -> str:
    context = (
        f"Product Name: {name}\n"
        f"Scraped Product Text: {scraped_text}\n"
        f"Manual Text: {manual_text}\n"
    )
    context = context[: cfg.max_input_chars]

    return (
        "Extract structured product information for a smart-home accessibility dataset.\n"
        "Include key product features and specifications from the <Manual Text> relevant to setup and everyday use.\n"
        f"Target description length: {cfg.target_words_min}-{cfg.target_words_max} words.\n"
        "Output must be valid JSON only with keys: product_name, description, category.\n"
        "Category should be hierarchical when possible, like Home > Climate > Thermostat.\n"
        "Do not include markdown or any extra keys.\n\n"
        f"{context}"
    )


def extract_json_blob(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object detected in LLM output")
    return json.loads(match.group(0))


def normalize_output_row(name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    product_name = str(payload.get("product_name", "") or "").strip() or name.strip()
    description = str(payload.get("description", "") or "").strip()
    category = str(payload.get("category", "") or "").strip()

    if not description:
        description = f"{name.strip()} smart-home product. Detailed product description unavailable."
    if not category:
        category = "Home > Smart Home Device"

    return {
        "product_name": product_name,
        "description": description,
        "category": category,
    }


def call_ollama(prompt: str, cfg: TransformConfig) -> Dict[str, Any]:
    client = OllamaClient(host=cfg.ollama_base_url)
    response = client.chat(
        model=cfg.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    content = response.get("message", {}).get("content", "")
    return extract_json_blob(content)


def call_groq(prompt: str, cfg: TransformConfig) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=cfg.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=cfg.llm_timeout,
    )
    content = completion.choices[0].message.content or ""
    return extract_json_blob(content)


def process_row(
    row: pd.Series,
    cfg: TransformConfig,
    logger: RunLogger,
    verbose: bool,
) -> Optional[Dict[str, str]]:
    name = str(row.get("name", "") or "").strip()
    product_url = str(row.get("product_url", "") or "").strip()
    manual_url = str(row.get("manual_url", "") or "").strip()

    if not name:
        logger.summary["failed"] += 1
        logger.add_row({"status": "failed", "error": "missing_name", "name": name})
        return None

    product_text = ""
    manual_text = ""

    try:
        product_text = fetch_url_text(product_url, cfg)
    except Exception as exc:
        logger.summary["scrape_failures"] += 1
        logger.add_row(
            {
                "status": "warning",
                "name": name,
                "url": product_url,
                "stage": "product_scrape",
                "error": str(exc),
            }
        )

    if manual_url:
        try:
            if row['manual_url'].endswith('.pdf'):
                resp = requests.get(row['manual_url'])
                with open('temp.pdf', 'wb') as f:
                    f.write(resp.content)
                from pypdf import PdfReader
                reader = PdfReader('temp.pdf')
                manual_text = ' '.join(page.extract_text() for page in reader.pages)[:3000]
                manual_text += f"\nManual: {manual_text}"
            os.remove('temp.pdf')
        except:
            pass  # Skip bad manuals
        try:
            manual_text = fetch_url_text(manual_url, cfg)
        except Exception as exc:
            logger.summary["scrape_failures"] += 1
            logger.add_row(
                {
                    "status": "warning",
                    "name": name,
                    "url": manual_url,
                    "stage": "manual_scrape",
                    "error": str(exc),
                }
            )

    if not product_text and not manual_text:
        product_text = name

    prompt = build_prompt(name, product_text, manual_text, cfg)

    try:
        extracted = call_ollama(prompt, cfg)
    except Exception as ollama_exc:
        logger.summary["fallback_to_groq"] += 1
        if verbose:
            print(f"[fallback->groq] {name}: {ollama_exc}")
        try:
            extracted = call_groq(prompt, cfg)
        except Exception as groq_exc:
            logger.summary["failed"] += 1
            logger.add_row(
                {
                    "status": "failed",
                    "name": name,
                    "error": "llm_extraction_failed",
                    "ollama_error": str(ollama_exc),
                    "groq_error": str(groq_exc),
                }
            )
            return None

    output_row = normalize_output_row(name, extracted)
    logger.summary["succeeded"] += 1
    logger.add_row({"status": "ok", "name": name})
    return output_row


def save_log(log_path: str, logger: RunLogger) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": logger.summary, "rows": logger.rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.timeout)
    logger = RunLogger()

    df_in = pd.read_csv(args.input)
    validate_input_schema(df_in)

    output_rows: List[Dict[str, str]] = []

    for _, row in df_in.iterrows():
        logger.summary["processed"] += 1
        out = process_row(row, cfg, logger, args.verbose)
        if out:
            output_rows.append(out)
        time.sleep(cfg.sleep_seconds)

    df_out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)

    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)

    save_log(args.log_path, logger)

    print(
        "Completed transform: "
        f"processed={logger.summary['processed']} "
        f"succeeded={logger.summary['succeeded']} "
        f"failed={logger.summary['failed']} "
        f"fallback_to_groq={logger.summary['fallback_to_groq']}"
    )


if __name__ == "__main__":
    main()
