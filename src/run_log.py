import threading
from typing import Dict, Any, List

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


