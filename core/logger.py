"""
AURAFORGE: Analytics Logger
Appends every analysis to a JSONL audit log.
Used for dashboard analytics, abuse detection, and model retraining data.

"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_logger = logging.getLogger("auraforge.analytics")


def log_analysis(result: dict):
    """
    Append one analysis result to logs/analyses.jsonl.
    Each line is a valid JSON object (JSONL format).
    
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    log_file = LOG_DIR / "analyses.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    _logger.info(
        f"[{result.get('decision','?')}] {result.get('file_name', 'unknown')} "
        f"→ {result.get('ai_probability', 0):.2%} AI probability"
    )
