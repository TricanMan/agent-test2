# LLM-as-Judge Prompt Testing MVP

This workspace provides a minimum viable pipeline to evaluate prompt variants against human ratings on smart home accessibility dimensions.

## What It Produces

- `outputs/llm_ratings.csv`: LLM scores and reasons by model/provider/prompt/product/dimension
- `outputs/metrics_table.csv`: Model+prompt correlations, rank correlations, accuracy, and kappa
- `outputs/summary_report.md`: Ranked summary with validation flags and scatter plots
- `outputs/logs.json`: Warnings, errors, token usage, and estimated API cost

## Inputs

1. Products CSV with columns:
   - `product_name`
   - `description`
   - `category`
   - optional `product_id` (auto-generated if missing)
2. Human ratings CSV with columns:
   - `product_id`, `rater_id`, `physical_score`, `cognitive_score`, `reason`
3. Prompt variants JSON list:
   - each item must include `name` and `template`
4. Config JSON (`config.json` by default)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cat << 'EOF' > .env
OPENAI_API_KEY="your_openai_key"
GOOGLE_API_KEY="your_google_key"
ANTHROPIC_API_KEY="your_anthropic_key"
EOF
```

## Run

```bash
python mvp.py --products sample/products.csv --human sample/human_ratings.csv --prompts prompts.json
```

Optional:

```bash
python mvp.py --products sample/products.csv --human sample/human_ratings.csv --prompts prompts.json --config config.json --output-dir outputs
```

## Notes

- The runner uses LiteLLM as a unified gateway across providers.
- Retries are enabled for LLM calls (`retry_attempts` in config).
- Invalid JSON or out-of-range model scores are logged and skipped.
- Low-agreement prompts are flagged when avg correlation < 0.4.
- Kappa uses a generalized Fleiss-style formulation over rating count matrices.
