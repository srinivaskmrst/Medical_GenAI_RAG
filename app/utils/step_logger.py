"""File-based step/timing logger for diagnosing slow RAG requests.

Writes one line per pipeline step to D:\\MedialChatbot_Log.txt so response-time
regressions can be traced to a specific stage (embedding, retrieval, LLM call, etc.)
without needing to reproduce the issue interactively.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

LOG_FILE = r"D:\MedialChatbot_Log.txt"

step_logger = logging.getLogger("medical_rag.steps")
step_logger.setLevel(logging.INFO)
if not step_logger.handlers:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    step_logger.addHandler(_handler)
    step_logger.propagate = False


@contextmanager
def log_step(request_id: str, label: str):
    """Log start/end + elapsed time for a named pipeline step."""
    start = time.perf_counter()
    step_logger.info(f"[{request_id}] START  {label}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        step_logger.info(f"[{request_id}] ERROR  {label} ({elapsed:.3f}s) -> {exc!r}")
        raise
    else:
        elapsed = time.perf_counter() - start
        step_logger.info(f"[{request_id}] DONE   {label} ({elapsed:.3f}s)")
