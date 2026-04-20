from typing import Any, Dict, Tuple
import json
import re

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