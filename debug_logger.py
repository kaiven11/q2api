"""
Debug logging utility for q2api 502 error investigation.
Provides structured logging with request context and detailed error information.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Configure structured logger
logger = logging.getLogger("q2api_debug")
logger.setLevel(logging.DEBUG)

# Log file lives next to the application by default so the path works both
# locally and inside the Docker image; allow overriding via env for flexibility.
base_dir = Path(os.getenv("Q2API_LOG_DIR", Path(__file__).resolve().parent))
log_file = base_dir / "debug_502.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

# Create file handler if not exists
if not logger.handlers:
    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_502_error(
    error_location: str,
    error_detail: str,
    request_id: Optional[str] = None,
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Log detailed information about 502 errors for debugging.

    Args:
        error_location: Where the error occurred (e.g., "token_refresh", "upstream_request")
        error_detail: The error message that will be returned to client
        request_id: Unique request identifier if available
        account_id: Account ID involved in the request
        api_key: API key used (masked for security)
        exception: The original exception if available
        context: Additional context information
    """

    # Mask API key for security
    masked_key = None
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_location": error_location,
        "error_detail": error_detail,
        "request_id": request_id,
        "account_id": account_id,
        "api_key_masked": masked_key,
        "context": context or {}
    }

    # Add exception details if available
    if exception:
        log_data["exception_type"] = type(exception).__name__
        log_data["exception_message"] = str(exception)
        log_data["traceback"] = traceback.format_exc()

    # Log as structured JSON
    logger.error(f"502_ERROR | {json.dumps(log_data, indent=None)}")

def log_request_context(
    request_id: str,
    endpoint: str,
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Log request context for successful requests to help with debugging patterns.
    """

    masked_key = None
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "account_id": account_id,
        "api_key_masked": masked_key,
        "additional_info": additional_info or {}
    }

    logger.info(f"REQUEST_CONTEXT | {json.dumps(log_data, indent=None)}")

def log_token_refresh_attempt(
    account_id: str,
    refresh_reason: str,
    context: Optional[Dict[str, Any]] = None
):
    """
    Log token refresh attempts to track refresh patterns.
    """

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "account_id": account_id,
        "refresh_reason": refresh_reason,
        "context": context or {}
    }

    logger.info(f"TOKEN_REFRESH | {json.dumps(log_data, indent=None)}")

def log_upstream_request_details(
    request_id: str,
    account_id: str,
    upstream_url: str,
    request_method: str,
    response_status: Optional[int] = None,
    response_time_ms: Optional[float] = None,
    error: Optional[str] = None
):
    """
    Log upstream request details for debugging connectivity issues.
    """

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "account_id": account_id,
        "upstream_url": upstream_url,
        "request_method": request_method,
        "response_status": response_status,
        "response_time_ms": response_time_ms,
        "error": error
    }

    logger.info(f"UPSTREAM_REQUEST | {json.dumps(log_data, indent=None)}")
