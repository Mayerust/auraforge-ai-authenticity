"""
AURAFORGE: API Key Authentication

"""

import os
from fastapi import HTTPException

#In production: load from database or secrets manager.
#Here, comma-separated keys in env var for simplicity.
VALID_API_KEYS = set(
    os.getenv("AURAFORGE_API_KEYS", "demo-key-123,test-key-456").split(",")
)


def validate_api_key(api_key: str | None):
    """
    Validate X-API-Key header.
    Raises HTTP 401 if missing, 403 if invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Add header: X-API-Key: <your-key>",
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. Contact admin@auraforge.io to register.",
        )
