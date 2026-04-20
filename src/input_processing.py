import pandas as pd
from src.run_log import RunLogger

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