import os
import json
import traceback
import uuid
import time
import asyncio
import importlib.util
import random
import secrets
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any, AsyncGenerator, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import Queue

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import tiktoken

from db import init_db, close_db, row_to_dict, DatabaseBackend, get_database_backend
from debug_logger import log_502_error, log_request_context, log_token_refresh_attempt, log_upstream_request_details

# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

try:
    # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

def count_tokens(text: str, apply_multiplier: bool = False) -> int:
    """Counts tokens with tiktoken."""
    if not text or not ENCODING:
        return 0
    token_count = len(ENCODING.encode(text))
    if apply_multiplier:
        token_count = int(token_count * TOKEN_COUNT_MULTIPLIER)
    return token_count

# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default

STREAM_PING_INTERVAL = max(0, _env_int("STREAM_PING_INTERVAL", 15))
FORCE_STREAM_ALL = _env_bool("FORCE_STREAM_ALL", False)
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 20)  # 限制历史消息数量

app = FastAPI(title="v2 OpenAI-compatible Server (Amazon Q Backend)")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS for simple testing in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Dynamic import of replicate.py to avoid package __init__ needs
# ------------------------------------------------------------------------------

def _load_replicate_module():
    mod_path = BASE_DIR / "replicate.py"
    spec = importlib.util.spec_from_file_location("v2_replicate", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_replicate = _load_replicate_module()
send_chat_request = _replicate.send_chat_request

# ------------------------------------------------------------------------------
# Dynamic import of Claude modules
# ------------------------------------------------------------------------------

def _load_claude_modules():
    # claude_types
    spec_types = importlib.util.spec_from_file_location("v2_claude_types", str(BASE_DIR / "claude_types.py"))
    mod_types = importlib.util.module_from_spec(spec_types)
    spec_types.loader.exec_module(mod_types)

    # claude_converter
    spec_conv = importlib.util.spec_from_file_location("v2_claude_converter", str(BASE_DIR / "claude_converter.py"))
    mod_conv = importlib.util.module_from_spec(spec_conv)
    # We need to inject claude_types into converter's namespace if it uses relative imports or expects them
    # But since we used relative import in claude_converter.py (.claude_types), we need to be careful.
    # Actually, since we are loading dynamically, relative imports might fail if not in sys.modules correctly.
    # Let's patch sys.modules temporarily or just rely on file location.
    # A simpler way for this single-file script style is to just load them.
    # However, claude_converter does `from .claude_types import ...`
    # To make that work, we should probably just use standard import if v2 is a package,
    # but v2 is just a folder.
    # Let's assume the user runs this with v2 in pythonpath or we just fix imports in the files.
    # But I wrote `from .claude_types` in the file.
    # Let's try to load it. If it fails, we might need to adjust.
    # Actually, for simplicity in this `app.py` dynamic loading context,
    # it is better if `claude_converter.py` used absolute import or we mock the package.
    # BUT, let's try to just load them and see.
    # To avoid relative import issues, I will inject the module into sys.modules
    import sys
    sys.modules["v2.claude_types"] = mod_types

    spec_conv.loader.exec_module(mod_conv)

    # claude_stream
    spec_stream = importlib.util.spec_from_file_location("v2_claude_stream", str(BASE_DIR / "claude_stream.py"))
    mod_stream = importlib.util.module_from_spec(spec_stream)
    spec_stream.loader.exec_module(mod_stream)

    # claude_websearch
    spec_websearch = importlib.util.spec_from_file_location("v2_claude_websearch", str(BASE_DIR / "claude_websearch.py"))
    mod_websearch = importlib.util.module_from_spec(spec_websearch)
    sys.modules["claude_types"] = mod_types  # Inject for websearch module
    spec_websearch.loader.exec_module(mod_websearch)

    return mod_types, mod_conv, mod_stream, mod_websearch

try:
    _claude_types, _claude_converter, _claude_stream, _claude_websearch = _load_claude_modules()
    ClaudeRequest = _claude_types.ClaudeRequest
    convert_claude_to_amazonq_request = _claude_converter.convert_claude_to_amazonq_request
    ClaudeStreamHandler = _claude_stream.ClaudeStreamHandler
    generate_thinking_prefix = getattr(_claude_converter, "generate_thinking_prefix", None)
    # Websearch functions
    has_web_search_tool = _claude_websearch.has_web_search_tool
    extract_search_query = _claude_websearch.extract_search_query
    create_mcp_request = _claude_websearch.create_mcp_request
    generate_websearch_events = _claude_websearch.generate_websearch_events
except Exception as e:
    print(f"Failed to load Claude modules: {e}")
    traceback.print_exc()
    # Define dummy classes to avoid NameError on startup if loading fails
    class ClaudeRequest(BaseModel):
        pass
    convert_claude_to_amazonq_request = None
    ClaudeStreamHandler = None
    generate_thinking_prefix = None
    has_web_search_tool = None
    extract_search_query = None
    create_mcp_request = None
    generate_websearch_events = None

# ------------------------------------------------------------------------------
# Global HTTP Client
# ------------------------------------------------------------------------------

GLOBAL_CLIENT: Optional[httpx.AsyncClient] = None

def _get_proxies() -> Optional[Dict[str, str]]:
    proxy = os.getenv("HTTP_PROXY", "").strip()
    if proxy:
        return {"http": proxy, "https": proxy}
    return None

async def _init_global_client():
    global GLOBAL_CLIENT
    proxies = _get_proxies()
    mounts = None
    if proxies:
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url:
            mounts = {
                "https://": httpx.AsyncHTTPTransport(proxy=proxy_url),
                "http://": httpx.AsyncHTTPTransport(proxy=proxy_url),
            }
    # Increased limits for high concurrency with streaming
    # max_connections: 总连接数上限
    # max_keepalive_connections: 保持活跃的连接数
    # keepalive_expiry: 连接保持时间
    limits = httpx.Limits(
        max_keepalive_connections=200,
        max_connections=500,  # 提高到500以支持更高并发
        keepalive_expiry=120.0  # 120秒后释放空闲连接 - 增加到 2 分钟
    )
    # 为流式响应设置更长的超时
    timeout = httpx.Timeout(
        connect=10.0,  # 连接超时
        read=1200.0,    # 读取超时(流式响应需要更长时间) - 增加到 20 分钟
        write=10.0,    # 写入超时
        pool=60.0      # 从连接池获取连接的超时时间(关键!)
    )
    GLOBAL_CLIENT = httpx.AsyncClient(mounts=mounts, timeout=timeout, limits=limits)

async def _close_global_client():
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT:
        await GLOBAL_CLIENT.aclose()
        GLOBAL_CLIENT = None

# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------

# Database backend instance (initialized on startup)
_db: Optional[DatabaseBackend] = None
_stats_queue = asyncio.Queue(maxsize=1024)
_stats_worker_task: Optional[asyncio.Task] = None

async def _ensure_db():
    """Initialize database backend."""
    global _db
    _db = await init_db()

def _row_to_dict(r: Dict[str, Any]) -> Dict[str, Any]:
    """Convert database row to dict with JSON parsing."""
    return row_to_dict(r)

# _ensure_db() will be called in startup event

# ------------------------------------------------------------------------------
# JWT Token Validation
# ------------------------------------------------------------------------------

def is_token_expired(access_token: str, buffer_seconds: int = 300) -> bool:
    """Check if JWT token is expired or will expire within buffer_seconds (default 5 min)"""
    try:
        parts = access_token.split('.')
        if len(parts) == 3:
            payload = base64.urlsafe_b64decode(parts[1] + '==')
            token_data = json.loads(payload)
            exp = token_data.get('exp')
            if exp:
                return time.time() >= (exp - buffer_seconds)
    except Exception:
        pass
    return False  # If we can't parse, assume valid (let backend handle it)

# ------------------------------------------------------------------------------
# Background token refresh thread
# ------------------------------------------------------------------------------

async def _refresh_stale_tokens():
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            if _db is None:
                print("[Error] Database not initialized, skipping token refresh cycle.")
                continue
            now = time.time()
            
            if LAZY_ACCOUNT_POOL_ENABLED:
                limit = LAZY_ACCOUNT_POOL_SIZE + LAZY_ACCOUNT_POOL_REFRESH_OFFSET
                order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
                query = f"SELECT id, last_refresh_time FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction} LIMIT {limit}"
                rows = await _db.fetchall(query)
            else:
                rows = await _db.fetchall("SELECT id, last_refresh_time FROM accounts WHERE enabled=1")

            for row in rows:
                acc_id, last_refresh = row['id'], row['last_refresh_time']
                should_refresh = False
                if not last_refresh or last_refresh == "never":
                    should_refresh = True
                else:
                    try:
                        last_time = time.mktime(time.strptime(last_refresh, "%Y-%m-%dT%H:%M:%S"))
                        if now - last_time > 300:  # 5 minutes
                            should_refresh = True
                    except Exception:
                        # Malformed or unparsable timestamp; force refresh
                        should_refresh = True

                if should_refresh:
                    try:
                        await refresh_access_token_in_db(acc_id)
                    except Exception:
                        traceback.print_exc()
                        # Ignore per-account refresh failure; timestamp/status are recorded inside
                        pass
        except Exception:
            traceback.print_exc()
            pass

# ------------------------------------------------------------------------------
# Env and API Key authorization (keys are independent of AWS accounts)
# ------------------------------------------------------------------------------
def _parse_key_list_env(env_name: str) -> List[str]:
    """
    Generic helper to parse comma-separated key lists from environment variables.
    """
    s = os.getenv(env_name, "") or ""
    return [k.strip() for k in s.split(",") if k.strip()]

ALLOWED_API_KEYS: List[str] = _parse_key_list_env("OPENAI_KEYS")
JSON_ONLY_API_KEYS: Set[str] = set(_parse_key_list_env("JSON_ONLY_API_KEYS"))
MAX_ERROR_COUNT: int = int(os.getenv("MAX_ERROR_COUNT", "100"))
TOKEN_COUNT_MULTIPLIER: float = float(os.getenv("TOKEN_COUNT_MULTIPLIER", "1.0"))
DEFAULT_CLAUDE_STREAM: bool = os.getenv("DEFAULT_CLAUDE_STREAM", "true").strip().lower() in ("true", "1", "yes")

# Lazy Account Pool settings
LAZY_ACCOUNT_POOL_ENABLED: bool = os.getenv("LAZY_ACCOUNT_POOL_ENABLED", "false").lower() in ("true", "1", "yes")
LAZY_ACCOUNT_POOL_SIZE: int = int(os.getenv("LAZY_ACCOUNT_POOL_SIZE", "20"))
LAZY_ACCOUNT_POOL_REFRESH_OFFSET: int = int(os.getenv("LAZY_ACCOUNT_POOL_REFRESH_OFFSET", "10"))
LAZY_ACCOUNT_POOL_ORDER_BY: str = os.getenv("LAZY_ACCOUNT_POOL_ORDER_BY", "created_at")
LAZY_ACCOUNT_POOL_ORDER_DESC: bool = os.getenv("LAZY_ACCOUNT_POOL_ORDER_DESC", "false").lower() in ("true", "1", "yes")

# Validate LAZY_ACCOUNT_POOL_ORDER_BY to prevent SQL injection
if LAZY_ACCOUNT_POOL_ORDER_BY not in ["created_at", "id", "success_count"]:
    LAZY_ACCOUNT_POOL_ORDER_BY = "created_at"

def _is_console_enabled() -> bool:
    """检查是否启用管理控制台"""
    console_env = os.getenv("ENABLE_CONSOLE", "true").strip().lower()
    return console_env not in ("false", "0", "no", "disabled")

CONSOLE_ENABLED: bool = _is_console_enabled()

# Admin authentication configuration
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "pknSask@0302")

CLAUDE_MODELS = [
    {
        "id": "claude-sonnet-4.5",
        "object": "model",
        "created": 1700000000,
        "owned_by": "amazon-q",
        "display_name": "Claude Sonnet 4.5",
    },
    {
        "id": "claude-opus-4.5",
        "object": "model",
        "created": 1700000000,
        "owned_by": "amazon-q",
        "display_name": "Claude Opus 4.5",
    },
    {
        "id": "claude-haiku-4.5",
        "object": "model",
        "created": 1700000000,
        "owned_by": "amazon-q",
        "display_name": "Claude Haiku 4.5",
    },
]

def _extract_bearer(token_header: Optional[str]) -> Optional[str]:
    if not token_header:
        return None
    if token_header.startswith("Bearer "):
        return token_header.split(" ", 1)[1].strip()
    return token_header.strip()

async def _list_enabled_accounts(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if LAZY_ACCOUNT_POOL_ENABLED:
        order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
        query = f"SELECT * FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction}"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    else:
        query = "SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    return [_row_to_dict(r) for r in rows]

async def _list_disabled_accounts() -> List[Dict[str, Any]]:
    rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=0 ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]

async def verify_account(account: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """验证账号可用性"""
    try:
        account = await refresh_access_token_in_db(account['id'])
        test_request = {
            "conversationState": {
                "currentMessage": {"userInputMessage": {"content": "hello"}},
                "chatTriggerType": "MANUAL"
            }
        }
        _, _, tracker, event_gen = await send_chat_request(
            access_token=account['accessToken'],
            messages=[],
            stream=True,
            raw_payload=test_request
        )
        if event_gen:
            async for _ in event_gen:
                break
        return True, None
    except Exception as e:
        if "AccessDenied" in str(e) or "403" in str(e):
            return False, "AccessDenied"
        return False, None

async def resolve_account_for_key(bearer_key: Optional[str]) -> Dict[str, Any]:
    """
    Authorize request by OPENAI_KEYS (if configured), then select an AWS account.
    Selection strategy: random among all enabled accounts. Authorization key does NOT map to any account.
    """
    # Authorization
    if ALLOWED_API_KEYS:
        if not bearer_key or bearer_key not in ALLOWED_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Selection: random among enabled accounts
    if LAZY_ACCOUNT_POOL_ENABLED:
        candidates = await _list_enabled_accounts(limit=LAZY_ACCOUNT_POOL_SIZE)
    else:
        candidates = await _list_enabled_accounts()

    if not candidates:
        raise HTTPException(status_code=401, detail="No enabled account available")
    return random.choice(candidates)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True

class BatchAccountCreate(BaseModel):
    accounts: List[AccountCreate]

class AccountUpdate(BaseModel):
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False

# ------------------------------------------------------------------------------
# Token refresh (OIDC)
# ------------------------------------------------------------------------------

OIDC_BASE = "https://oidc.us-east-1.amazonaws.com"
TOKEN_URL = f"{OIDC_BASE}/token"

def _oidc_headers() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "user-agent": "aws-sdk-rust/1.3.9 os/windows lang/rust/1.87.0",
        "x-amz-user-agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/windows lang/rust/1.87.0 m/E app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }

async def refresh_access_token_in_db(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    acc = _row_to_dict(row)

    if not acc.get("clientId") or not acc.get("clientSecret") or not acc.get("refreshToken"):
        raise HTTPException(status_code=400, detail="Account missing clientId/clientSecret/refreshToken for refresh")

    payload = {
        "grantType": "refresh_token",
        "clientId": acc["clientId"],
        "clientSecret": acc["clientSecret"],
        "refreshToken": acc["refreshToken"],
    }

    try:
        # Use global client if available, else fallback (though global should be ready)
        client = GLOBAL_CLIENT
        if not client:
            # Fallback for safety
            async with httpx.AsyncClient(timeout=60.0) as temp_client:
                r = await temp_client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
                r.raise_for_status()
                data = r.json()
        else:
            r = await client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        new_access = data.get("accessToken")
        new_refresh = data.get("refreshToken", acc.get("refreshToken"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "success"
    except httpx.HTTPError as e:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        # 记录刷新失败次数
        await _update_stats(account_id, False)
        # Log detailed 502 error information
        log_502_error(
            error_location="token_refresh_http_error",
            error_detail=f"Token refresh failed: {str(e)}",
            account_id=str(account_id),
            exception=e,
            context={"refresh_attempt": "OIDC token refresh", "error_type": "httpx.HTTPError"}
        )
        raise HTTPException(status_code=502, detail=f"Token refresh failed: {str(e)}")
    except Exception as e:
        # Ensure last_refresh_time is recorded even on unexpected errors
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        # 记录刷新失败次数
        await _update_stats(account_id, False)
        raise

    await _db.execute(
        """
        UPDATE accounts
        SET accessToken=?, refreshToken=?, last_refresh_time=?, last_refresh_status=?, updated_at=?
        WHERE id=?
        """,
        (new_access, new_refresh, now, status, now, account_id),
    )

    row2 = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    return _row_to_dict(row2)

async def get_account(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return _row_to_dict(row)

async def ensure_valid_token(account: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure account has a valid token, refresh if expired or expiring soon"""
    access_token = account.get("accessToken")

    # Quick check: if token was refreshed recently (< 10 min), skip JWT parsing
    last_refresh = account.get("last_refresh_time")
    if last_refresh and last_refresh != "never":
        try:
            last_time = time.mktime(time.strptime(last_refresh, "%Y-%m-%dT%H:%M:%S"))
            if time.time() - last_time < 600:  # Refreshed within 10 minutes
                return account
        except Exception:
            pass

    # Only parse JWT if token might be expiring soon
    if not access_token or is_token_expired(access_token):
        account = await refresh_access_token_in_db(account["id"])

    return account

async def mark_account_banned(account_id: str, reason: str = "TEMPORARILY_SUSPENDED") -> None:
    """Mark account as banned and disabled"""
    if not _db:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    await _db.execute(
        "UPDATE accounts SET enabled=0, updated_at=? WHERE id=?",
        (now, account_id),
    )
    print(f"[Ban Detected] Account {account_id} marked as banned: {reason}")

async def _apply_stats_update(account_id: str, success: bool) -> None:
    """Apply stats updates; executed by worker to avoid DB contention."""
    if not _db:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    if success:
        await _db.execute(
            "UPDATE accounts SET success_count=success_count+1, error_count=0, updated_at=? WHERE id=?",
            (now, account_id),
        )
        return

    row = await _db.fetchone("SELECT error_count FROM accounts WHERE id=?", (account_id,))
    if not row:
        return
    new_count = (row["error_count"] or 0) + 1
    if new_count >= MAX_ERROR_COUNT:
        await _db.execute(
            "UPDATE accounts SET error_count=?, enabled=0, updated_at=? WHERE id=?",
            (new_count, now, account_id),
        )
    else:
        await _db.execute(
            "UPDATE accounts SET error_count=?, updated_at=? WHERE id=?",
            (new_count, now, account_id),
        )


async def _update_stats(account_id: str, success: bool) -> None:
    """Queue stats update; fall back to inline update when necessary."""
    worker_ready = _stats_worker_task is not None and not _stats_worker_task.done()
    if not worker_ready:
        await _apply_stats_update(account_id, success)
        return

    try:
        await asyncio.wait_for(_stats_queue.put((account_id, success)), timeout=0.5)
    except asyncio.TimeoutError:
        await _apply_stats_update(account_id, success)


async def _stats_worker() -> None:
    """Background worker that serializes stats updates."""
    while True:
        try:
            account_id, success = await _stats_queue.get()
        except asyncio.CancelledError:
            break
        try:
            await _apply_stats_update(account_id, success)
        except Exception:
            traceback.print_exc()
        finally:
            _stats_queue.task_done()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

async def require_account(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    key = _extract_bearer(authorization) if authorization else x_api_key
    account = await resolve_account_for_key(key)
    if key and key in JSON_ONLY_API_KEYS:
        account["_force_json_response"] = True
    else:
        account.pop("_force_json_response", None)
    return account

def verify_admin_password(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin password for console access"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )

    password = authorization[7:]  # Remove "Bearer " prefix

    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid password", "code": "INVALID_PASSWORD"}
        )

    return True

# ------------------------------------------------------------------------------
# OpenAI-compatible Chat endpoint
# ------------------------------------------------------------------------------

def _openai_non_streaming_response(
    text: str,
    model: Optional[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

def _sse_format(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

async def handle_websearch_request(req: ClaudeRequest, account: Dict[str, Any]) -> StreamingResponse:
    """Handle websearch-only requests."""
    query = extract_search_query(req)
    if not query:
        raise HTTPException(status_code=400, detail="Cannot extract search query from request")

    # Estimate input tokens
    input_tokens = count_tokens(query) + 100

    # Create MCP request
    tool_use_id, mcp_request = create_mcp_request(query)

    # Try to call MCP API (Amazon Q MCP endpoint)
    search_results = None
    try:
        account = await ensure_valid_token(account)
        access_token = account.get("accessToken")
        if access_token and GLOBAL_CLIENT:
            mcp_url = "https://q.us-east-1.amazonaws.com/mcp"
            headers = {
                "authorization": f"Bearer {access_token}",
                "content-type": "application/json"
            }
            response = await GLOBAL_CLIENT.post(mcp_url, json=mcp_request, headers=headers, timeout=30.0)
            if response.status_code == 200:
                mcp_response = response.json()
                if mcp_response.get("result") and mcp_response["result"].get("content"):
                    content = mcp_response["result"]["content"][0]
                    if content.get("type") == "text":
                        search_results = json.loads(content["text"])
    except Exception as e:
        print(f"[WebSearch] MCP API call failed: {e}")

    # Generate SSE events
    events = generate_websearch_events(req.model, query, tool_use_id, search_results, input_tokens)

    async def event_generator():
        for event in events:
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/v1/models")
async def list_models():
    """Return available Claude-compatible models."""
    return {
        "object": "list",
        "data": CLAUDE_MODELS,
    }

@app.post("/v1/messages")
async def claude_messages(
    req: ClaudeRequest,
    account: Dict[str, Any] = Depends(require_account),
    accept: Optional[str] = Header(default=None),
):
    """
    Claude-compatible messages endpoint.
    """
    # Check for websearch request first
    if has_web_search_tool and has_web_search_tool(req):
        return await handle_websearch_request(req, account)

    fields_set = getattr(req, "model_fields_set", getattr(req, "__fields_set__", set()))
    stream_field_present = isinstance(fields_set, set) and ("stream" in fields_set)
    accept_header = (accept or "").lower()
    wants_sse = "text/event-stream" in accept_header
    if stream_field_present:
        use_stream = bool(req.stream)
    else:
        # 当客户端未提供 stream 字段时，根据 Accept 和默认配置判定
        use_stream = DEFAULT_CLAUDE_STREAM and wants_sse

    if account.get("_force_json_response"):
        use_stream = False
    elif FORCE_STREAM_ALL:
        # 当 env 强制打开时，总是进入流式
        use_stream = True

    # 1. Convert request
    try:
        aq_request = convert_claude_to_amazonq_request(req, max_history=MAX_HISTORY_MESSAGES)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request conversion failed: {str(e)}")

    stats_recorded = False
    generator_manages_stats = False
    event_iter = None
    try:
        # Ensure token is valid before making request
        account = await ensure_valid_token(account)
        access = account.get("accessToken")
        if not access:
            stats_recorded = True
            log_502_error(
                error_location="access_token_unavailable_after_refresh",
                error_detail="Access token unavailable after refresh",
                account_id=str(account["id"]),
                context={"refresh_attempted": True, "account_status": account.get("enabled", "unknown")}
            )
            raise HTTPException(status_code=502, detail="Access token unavailable after refresh")

        # Always stream upstream so we can forward tokens immediately
        try:
            _, _, tracker, event_iter = await send_chat_request(
                access_token=access,
                messages=[],
                model=req.model,
                stream=True,
                client=GLOBAL_CLIENT,
                raw_payload=aq_request
            )
        except Exception as e:
            # If 403 error, try to refresh token and retry once
            if "403" in str(e):
                print(f"[403 Error] Account {account.get('id')} ({account.get('label')}) got 403, refreshing token...")
                try:
                    refreshed = await refresh_access_token_in_db(account["id"])
                    access = refreshed.get("accessToken")
                    if access:
                        _, _, tracker, event_iter = await send_chat_request(
                            access_token=access,
                            messages=[],
                            model=req.model,
                            stream=True,
                            client=GLOBAL_CLIENT,
                            raw_payload=aq_request
                        )
                        print(f"[403 Retry] Account {account.get('id')} retry succeeded after token refresh")
                    else:
                        stats_recorded = True
                        print(f"[403 Failed] Account {account.get('id')} token refresh returned empty")
                        await mark_account_banned(account["id"], "403_TOKEN_REFRESH_FAILED")
                        # Log detailed 502 error information
                        log_502_error(
                            error_location="access_token_refresh_failed_403_retry",
                            error_detail="Access token refresh failed",
                            account_id=str(account["id"]),
                            context={"retry_after_403": True, "refresh_returned_empty": True}
                        )
                        raise HTTPException(status_code=502, detail="Access token refresh failed")
                except HTTPException:
                    raise
                except Exception as retry_error:
                    stats_recorded = True
                    print(f"[403 Failed] Account {account.get('id')} retry failed: {str(retry_error)}")
                    if "403" in str(retry_error):
                        await mark_account_banned(account["id"], "403_RETRY_FAILED")
                    # Log detailed 502 error information
                    log_502_error(
                        error_location="request_failed_after_token_refresh",
                        error_detail=f"Request failed after token refresh: {str(e)}",
                        account_id=str(account["id"]),
                        exception=retry_error,
                        context={"retry_after_403": True, "original_error": str(e)}
                    )
                    raise HTTPException(status_code=502, detail=f"Request failed after token refresh: {str(e)}")
            else:
                stats_recorded = True
                error_str = str(e)
                # Check for account ban/suspension
                if "TEMPORARILY_SUSPENDED" in error_str or "suspended" in error_str.lower():
                    await mark_account_banned(account["id"], error_str)
                # Log detailed 502 error information
                log_502_error(
                    error_location="upstream_request_failed",
                    error_detail=f"Upstream request failed: {error_str}",
                    account_id=str(account["id"]),
                    exception=e,
                    context={"request_type": "send_chat_request", "model": req.model}
                )
                raise HTTPException(status_code=502, detail=f"Upstream request failed: {error_str}")

        if not event_iter:
            stats_recorded = True
            # Log detailed 502 error information
            log_502_error(
                error_location="no_event_stream_returned",
                error_detail="No event stream returned",
                account_id=str(account["id"]),
                context={"request_type": "send_chat_request", "model": req.model}
            )
            raise HTTPException(status_code=502, detail="No event stream returned")

        # Handler
        text_to_count = ""
        if req.system:
            if isinstance(req.system, str):
                text_to_count += req.system
            elif isinstance(req.system, list):
                for item in req.system:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_to_count += item.get("text", "")

        for msg in req.messages:
            if isinstance(msg.content, str):
                text_to_count += msg.content
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_to_count += item.get("text", "")

        input_tokens = count_tokens(text_to_count, apply_multiplier=True)
        thinking_enabled = bool(
            getattr(req, "thinking", None)
            and getattr(req.thinking, "thinking_type", "").lower() == "enabled"
        )
        handler = ClaudeStreamHandler(
            model=req.model,
            input_tokens=input_tokens,
            thinking_enabled=thinking_enabled,
        )
        response_message_id = handler.message_id

        # Fetch first event to surface upstream errors before we start streaming
        first_event = None
        try:
            first_event = await event_iter.__anext__()
        except StopAsyncIteration:
            stats_recorded = True
            # Log detailed 502 error information
            log_502_error(
                error_location="empty_response_from_upstream",
                error_detail="Empty response from upstream",
                account_id=str(account["id"]),
                context={"request_type": "event_stream", "model": req.model}
            )
            raise HTTPException(status_code=502, detail="Empty response from upstream")
        except Exception as e:
            stats_recorded = True
            # Log detailed 502 error information
            log_502_error(
                error_location="upstream_error_during_stream",
                error_detail=f"Upstream error: {str(e)}",
                account_id=str(account["id"]),
                exception=e,
                context={"request_type": "event_stream", "model": req.model}
            )
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

        async def event_generator(enable_pings: bool):
            """
            Relay upstream SSE events and optionally insert periodic ping events to keep the
            connection alive. A queue is used so that ping events can interleave without blocking
            downstream consumption.
            """
            stop_marker = object()
            queue: asyncio.Queue = asyncio.Queue()
            stream_error: Dict[str, Optional[BaseException]] = {"exc": None}

            async def stream_worker():
                try:
                    if first_event:
                        event_type, payload = first_event
                        async for sse in handler.handle_event(event_type, payload):
                            await queue.put(sse)

                    async for event_type, payload in event_iter:
                        async for sse in handler.handle_event(event_type, payload):
                            await queue.put(sse)
                    async for sse in handler.finish():
                        await queue.put(sse)
                    await _update_stats(account["id"], True)
                except asyncio.CancelledError:
                    await _update_stats(account["id"], tracker.has_content if tracker else False)
                    raise
                except Exception as exc:
                    stream_error["exc"] = exc
                    await _update_stats(account["id"], False)
                    raise
                finally:
                    await queue.put(stop_marker)

            async def ping_worker():
                try:
                    while True:
                        await asyncio.sleep(STREAM_PING_INTERVAL)
                        if STREAM_PING_INTERVAL <= 0:
                            break
                        await queue.put(build_ping())
                except asyncio.CancelledError:
                    pass

            producers = [asyncio.create_task(stream_worker())]
            if enable_pings and STREAM_PING_INTERVAL > 0:
                producers.append(asyncio.create_task(ping_worker()))

            try:
                while True:
                    item = await queue.get()
                    if item is stop_marker:
                        break
                    yield item
            finally:
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
                if stream_error["exc"]:
                    raise stream_error["exc"]

        generator_manages_stats = True

        if use_stream:
            return StreamingResponse(
                event_generator(enable_pings=True),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
                }
            )
        else:
            usage = {"input_tokens": 0, "output_tokens": 0}
            stop_reason = None
            final_content = []

            async for sse_chunk in event_generator(enable_pings=False):
                data_str = None
                for line in sse_chunk.strip().split('\n'):
                    if line.startswith("data:"):
                        data_str = line[6:].strip()
                        break

                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                    dtype = data.get("type")

                    if dtype == "content_block_start":
                        idx = data.get("index", 0)
                        while len(final_content) <= idx:
                            final_content.append(None)
                        final_content[idx] = data.get("content_block")

                    elif dtype == "content_block_delta":
                        idx = data.get("index", 0)
                        delta = data.get("delta", {})
                        if final_content[idx]:
                            if delta.get("type") == "text_delta":
                                final_content[idx]["text"] += delta.get("text", "")
                            elif delta.get("type") == "input_json_delta":
                                if "partial_json" not in final_content[idx]:
                                    final_content[idx]["partial_json"] = ""
                                final_content[idx]["partial_json"] += delta.get("partial_json", "")
                            elif delta.get("type") == "thinking_delta":
                                if "thinking" not in final_content[idx]:
                                    final_content[idx]["thinking"] = ""
                                final_content[idx]["thinking"] += delta.get("thinking", "")

                    elif dtype == "content_block_stop":
                        idx = data.get("index", 0)
                        if final_content[idx] and final_content[idx].get("type") == "tool_use":
                            if "partial_json" in final_content[idx]:
                                try:
                                    final_content[idx]["input"] = json.loads(final_content[idx]["partial_json"])
                                except json.JSONDecodeError:
                                    final_content[idx]["input"] = {"error": "invalid json", "partial": final_content[idx]["partial_json"]}
                                del final_content[idx]["partial_json"]

                    elif dtype == "message_delta":
                        usage = data.get("usage", usage)
                        stop_reason = data.get("delta", {}).get("stop_reason")

                except json.JSONDecodeError:
                    pass
                except Exception:
                    traceback.print_exc()
                    pass

            final_content_cleaned = []
            for c in final_content:
                if c is not None:
                    c.pop("partial_json", None)
                    final_content_cleaned.append(c)

            final_response_id = response_message_id or f"msg_{uuid.uuid4()}"
            return JSONResponse(content={
                "id": final_response_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": final_content_cleaned,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": usage
            })

    except Exception:
        try:
            if event_iter and hasattr(event_iter, "aclose"):
                await event_iter.aclose()
        except Exception:
            pass
        if not generator_manages_stats and not stats_recorded:
            await _update_stats(account["id"], False)
        raise

@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(req: ClaudeRequest):
    """
    Count tokens in a message without sending it.
    Compatible with Claude API's /v1/messages/count_tokens endpoint.
    Uses tiktoken for local token counting.
    """
    text_to_count = ""
    thinking_prefix = None
    if getattr(req, "thinking", None):
        if generate_thinking_prefix:
            try:
                thinking_prefix = generate_thinking_prefix(req.thinking)
            except Exception:
                thinking_prefix = None
        if not thinking_prefix and getattr(req.thinking, "thinking_type", "").lower() == "enabled":
            budget = getattr(req.thinking, "budget_tokens", 20000) or 20000
            thinking_prefix = (
                "<thinking_mode>enabled</thinking_mode>"
                f"<max_thinking_length>{int(budget)}</max_thinking_length>"
            )
    if thinking_prefix:
        text_to_count += thinking_prefix
    
    # Count system prompt tokens
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count message tokens
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count tool definition tokens if present
    if req.tools:
        text_to_count += json.dumps([tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in req.tools], ensure_ascii=False)
    
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)
    
    return {"input_tokens": input_tokens}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, account: Dict[str, Any] = Depends(require_account)):
    """
    OpenAI-compatible chat endpoint.
    - stream default False
    - messages will be converted into "{role}:\n{content}" and injected into template
    - account is chosen randomly among enabled accounts (API key is for authorization only)
    """
    model = req.model
    do_stream = bool(req.stream)

    async def _send_upstream(stream: bool) -> Tuple[Optional[str], Optional[AsyncGenerator[str, None]], Any]:
        access = account.get("accessToken")
        if not access:
            refreshed = await refresh_access_token_in_db(account["id"])
            access = refreshed.get("accessToken")
            if not access:
                # Log detailed 502 error information
                log_502_error(
                    error_location="access_token_unavailable_after_refresh_openai",
                    error_detail="Access token unavailable after refresh",
                    account_id=str(account["id"]),
                    context={"endpoint": "openai_chat_completions", "refresh_attempted": True}
                )
                raise HTTPException(status_code=502, detail="Access token unavailable after refresh")
        # Note: send_chat_request signature changed, but we use keyword args so it should be fine if we don't pass raw_payload
        # But wait, the return signature changed too! It now returns 4 values.
        # We need to unpack 4 values.
        result = await send_chat_request(access, [m.model_dump() for m in req.messages], model=model, stream=stream, client=GLOBAL_CLIENT)
        return result[0], result[1], result[2] # Ignore the 4th value (event_stream) for OpenAI endpoint

    if not do_stream:
        try:
            # Calculate prompt tokens
            prompt_text = "".join([m.content for m in req.messages if isinstance(m.content, str)])
            prompt_tokens = count_tokens(prompt_text)

            text, _, tracker = await _send_upstream(stream=False)
            await _update_stats(account["id"], bool(text))
            
            completion_tokens = count_tokens(text or "")
            
            return JSONResponse(content=_openai_non_streaming_response(
                text or "",
                model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            ))
        except Exception as e:
            await _update_stats(account["id"], False)
            raise
    else:
        created = int(time.time())
        stream_id = f"chatcmpl-{uuid.uuid4()}"
        model_used = model or "unknown"
        
        it = None
        try:
            # Calculate prompt tokens
            prompt_text = "".join([m.content for m in req.messages if isinstance(m.content, str)])
            prompt_tokens = count_tokens(prompt_text)

            _, it, tracker = await _send_upstream(stream=True)
            assert it is not None
            
            async def event_gen() -> AsyncGenerator[str, None]:
                completion_text = ""
                try:
                    # Send role first
                    yield _sse_format({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_used,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    })
                    
                    # Stream content
                    async for piece in it:
                        if piece:
                            completion_text += piece
                            yield _sse_format({
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_used,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            })
                    
                    # Send stop and usage
                    completion_tokens = count_tokens(completion_text)
                    yield _sse_format({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_used,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                    })
                    
                    yield "data: [DONE]\n\n"
                    await _update_stats(account["id"], True)
                except GeneratorExit:
                    # Client disconnected - update stats but don't re-raise
                    await _update_stats(account["id"], tracker.has_content if tracker else False)
                except Exception:
                    await _update_stats(account["id"], tracker.has_content if tracker else False)
                    raise
            
            return StreamingResponse(event_gen(), media_type="text/event-stream")
        except Exception as e:
            # Ensure iterator (if created) is closed to release upstream connection
            try:
                if it and hasattr(it, "aclose"):
                    await it.aclose()
            except Exception:
                pass
            await _update_stats(account["id"], False)
            raise

# ------------------------------------------------------------------------------
# Device Authorization (URL Login, 5-minute timeout)
# ------------------------------------------------------------------------------

# Dynamic import of auth_flow.py (device-code login helpers)
def _load_auth_flow_module():
    mod_path = BASE_DIR / "auth_flow.py"
    spec = importlib.util.spec_from_file_location("v2_auth_flow", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_auth_flow = _load_auth_flow_module()
register_client_min = _auth_flow.register_client_min
device_authorize = _auth_flow.device_authorize
poll_token_device_code = _auth_flow.poll_token_device_code

# In-memory auth sessions (ephemeral)
AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}

class AuthStartBody(BaseModel):
    label: Optional[str] = None
    enabled: Optional[bool] = True

class AdminLoginRequest(BaseModel):
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    message: str

async def _create_account_from_tokens(
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: Optional[str],
    label: Optional[str],
    enabled: bool,
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    acc_id = str(uuid.uuid4())
    await _db.execute(
        """
        INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            acc_id,
            label,
            client_id,
            client_secret,
            refresh_token,
            access_token,
            None,
            now,
            "success",
            now,
            now,
            1 if enabled else 0,
        ),
    )
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
    return _row_to_dict(row)

# 管理控制台相关端点 - 仅在启用时注册
if CONSOLE_ENABLED:
    # ------------------------------------------------------------------------------
    # Admin Authentication Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/api/login", response_model=AdminLoginResponse)
    async def admin_login(request: AdminLoginRequest) -> AdminLoginResponse:
        """Admin login endpoint - password only"""
        if request.password == ADMIN_PASSWORD:
            return AdminLoginResponse(
                success=True,
                message="Login successful"
            )
        else:
            return AdminLoginResponse(
                success=False,
                message="Invalid password"
            )

    @app.get("/login", response_class=FileResponse)
    def login_page():
        """Serve the login page"""
        path = BASE_DIR / "frontend" / "login.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/login.html not found")
        return FileResponse(str(path))

    # ------------------------------------------------------------------------------
    # Device Authorization Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/v2/auth/start")
    async def auth_start(body: AuthStartBody, _: bool = Depends(verify_admin_password)):
        """
        Start device authorization and return verification URL for user login.
        Session lifetime capped at 5 minutes on claim.
        """
        try:
            cid, csec = await register_client_min()
            dev = await device_authorize(cid, csec)
        except httpx.HTTPError as e:
            # Log detailed 502 error information
            log_502_error(
                error_location="oidc_device_authorize_error",
                error_detail=f"OIDC error: {str(e)}",
                exception=e,
                context={"operation": "device_authorize", "step": "register_client_and_device_auth"}
            )
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

        auth_id = str(uuid.uuid4())
        sess = {
            "clientId": cid,
            "clientSecret": csec,
            "deviceCode": dev.get("deviceCode"),
            "interval": int(dev.get("interval", 1)),
            "expiresIn": int(dev.get("expiresIn", 600)),
            "verificationUriComplete": dev.get("verificationUriComplete"),
            "userCode": dev.get("userCode"),
            "startTime": int(time.time()),
            "label": body.label,
            "enabled": True if body.enabled is None else bool(body.enabled),
            "status": "pending",
            "error": None,
            "accountId": None,
        }
        AUTH_SESSIONS[auth_id] = sess
        return {
            "authId": auth_id,
            "verificationUriComplete": sess["verificationUriComplete"],
            "userCode": sess["userCode"],
            "expiresIn": sess["expiresIn"],
            "interval": sess["interval"],
        }

    @app.get("/v2/auth/status/{auth_id}")
    async def auth_status(auth_id: str, _: bool = Depends(verify_admin_password)):
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        now_ts = int(time.time())
        deadline = sess["startTime"] + min(int(sess.get("expiresIn", 600)), 300)
        remaining = max(0, deadline - now_ts)
        return {
            "status": sess.get("status"),
            "remaining": remaining,
            "error": sess.get("error"),
            "accountId": sess.get("accountId"),
        }

    @app.post("/v2/auth/claim/{auth_id}")
    async def auth_claim(auth_id: str, _: bool = Depends(verify_admin_password)):
        """
        Block up to 5 minutes to exchange the device code for tokens after user completed login.
        On success, creates an enabled account and returns it.
        """
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        if sess.get("status") in ("completed", "timeout", "error"):
            return {
                "status": sess["status"],
                "accountId": sess.get("accountId"),
                "error": sess.get("error"),
            }
        try:
            toks = await poll_token_device_code(
                sess["clientId"],
                sess["clientSecret"],
                sess["deviceCode"],
                sess["interval"],
                sess["expiresIn"],
                max_timeout_sec=300,  # 5 minutes
            )
            access_token = toks.get("accessToken")
            refresh_token = toks.get("refreshToken")
            if not access_token:
                # Log detailed 502 error information
                log_502_error(
                    error_location="no_access_token_from_oidc",
                    error_detail="No accessToken returned from OIDC",
                    context={"operation": "poll_token_device_code", "tokens_received": bool(toks)}
                )
                raise HTTPException(status_code=502, detail="No accessToken returned from OIDC")

            acc = await _create_account_from_tokens(
                sess["clientId"],
                sess["clientSecret"],
                access_token,
                refresh_token,
                sess.get("label"),
                sess.get("enabled", True),
            )
            sess["status"] = "completed"
            sess["accountId"] = acc["id"]
            return {
                "status": "completed",
                "account": acc,
            }
        except TimeoutError:
            sess["status"] = "timeout"
            raise HTTPException(status_code=408, detail="Authorization timeout (5 minutes)")
        except httpx.HTTPError as e:
            sess["status"] = "error"
            sess["error"] = str(e)
            # Log detailed 502 error information
            log_502_error(
                error_location="oidc_poll_token_error",
                error_detail=f"OIDC error: {str(e)}",
                exception=e,
                context={"operation": "poll_token_device_code", "session_status": "error"}
            )
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

    # ------------------------------------------------------------------------------
    # Accounts Management API
    # ------------------------------------------------------------------------------

    @app.post("/v2/accounts")
    async def create_account(body: AccountCreate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        acc_id = str(uuid.uuid4())
        other_str = json.dumps(body.other, ensure_ascii=False) if body.other is not None else None
        enabled_val = 1 if (body.enabled is None or body.enabled) else 0
        await _db.execute(
            """
            INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acc_id,
                body.label,
                body.clientId,
                body.clientSecret,
                body.refreshToken,
                body.accessToken,
                other_str,
                None,
                "never",
                now,
                now,
                enabled_val,
            ),
        )
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
        return _row_to_dict(row)


    async def _verify_and_enable_accounts(account_ids: List[str]):
        """后台异步验证并启用账号"""
        for acc_id in account_ids:
            try:
                # 必须先获取完整的账号信息
                account = await get_account(acc_id)
                verify_success, fail_reason = await verify_account(account)
                now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

                if verify_success:
                    await _db.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, acc_id))
                elif fail_reason:
                    other_dict = account.get("other", {}) or {}
                    other_dict['failedReason'] = fail_reason
                    await _db.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, acc_id))
            except Exception as e:
                print(f"Error verifying account {acc_id}: {e}")
                traceback.print_exc()

    @app.post("/v2/accounts/feed")
    async def create_accounts_feed(request: BatchAccountCreate, _: bool = Depends(verify_admin_password)):
        """
        统一的投喂接口，接收账号列表，立即存入并后台异步验证。
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        new_account_ids = []

        for i, account_data in enumerate(request.accounts):
            acc_id = str(uuid.uuid4())
            other_dict = account_data.other or {}
            other_dict['source'] = 'feed'
            other_str = json.dumps(other_dict, ensure_ascii=False)

            await _db.execute(
                """
                INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    acc_id,
                    account_data.label or f"批量账号 {i+1}",
                    account_data.clientId,
                    account_data.clientSecret,
                    account_data.refreshToken,
                    account_data.accessToken,
                    other_str,
                    None,
                    "never",
                    now,
                    now,
                    0,  # 初始为禁用状态
                ),
            )
            new_account_ids.append(acc_id)

        # 启动后台任务进行验证，不阻塞当前请求
        if new_account_ids:
            asyncio.create_task(_verify_and_enable_accounts(new_account_ids))

        return {
            "status": "processing",
            "message": f"{len(new_account_ids)} accounts received and are being verified in the background.",
            "account_ids": new_account_ids
        }

    @app.get("/v2/accounts")
    async def list_accounts(_: bool = Depends(verify_admin_password), enabled: Optional[bool] = None, sort_by: str = "created_at", sort_order: str = "desc"):
        query = "SELECT * FROM accounts"
        params = []
        if enabled is not None:
            query += " WHERE enabled=?"
            params.append(1 if enabled else 0)
        sort_field = "created_at" if sort_by not in ["created_at", "success_count"] else sort_by
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        query += f" ORDER BY {sort_field} {order}"
        rows = await _db.fetchall(query, tuple(params) if params else ())
        accounts = [_row_to_dict(r) for r in rows]
        return {"accounts": accounts, "count": len(accounts)}

    @app.get("/v2/accounts/{account_id}")
    async def get_account_detail(account_id: str, _: bool = Depends(verify_admin_password)):
        return await get_account(account_id)

    @app.delete("/v2/accounts/{account_id}")
    async def delete_account(account_id: str, _: bool = Depends(verify_admin_password)):
        rowcount = await _db.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        return {"deleted": account_id}

    @app.patch("/v2/accounts/{account_id}")
    async def update_account(account_id: str, body: AccountUpdate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        fields = []
        values: List[Any] = []

        if body.label is not None:
            fields.append("label=?"); values.append(body.label)
        if body.clientId is not None:
            fields.append("clientId=?"); values.append(body.clientId)
        if body.clientSecret is not None:
            fields.append("clientSecret=?"); values.append(body.clientSecret)
        if body.refreshToken is not None:
            fields.append("refreshToken=?"); values.append(body.refreshToken)
        if body.accessToken is not None:
            fields.append("accessToken=?"); values.append(body.accessToken)
        if body.other is not None:
            fields.append("other=?"); values.append(json.dumps(body.other, ensure_ascii=False))
        if body.enabled is not None:
            fields.append("enabled=?"); values.append(1 if body.enabled else 0)

        if not fields:
            return await get_account(account_id)

        fields.append("updated_at=?"); values.append(now)
        values.append(account_id)

        rowcount = await _db.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id=?", tuple(values))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
        return _row_to_dict(row)

    @app.post("/v2/accounts/{account_id}/refresh")
    async def manual_refresh(account_id: str, _: bool = Depends(verify_admin_password)):
        return await refresh_access_token_in_db(account_id)

    # ------------------------------------------------------------------------------
    # Simple Frontend (minimal dev test page; full UI in v2/frontend/index.html)
    # ------------------------------------------------------------------------------

    # Frontend inline HTML removed; serving ./frontend/index.html instead (see route below)
    # Note: This route is NOT protected - the HTML file is served freely,
    # but the frontend JavaScript checks authentication and redirects to /login if needed.
    # All API endpoints remain protected.

    @app.get("/", response_class=FileResponse)
    def index():
        path = BASE_DIR / "frontend" / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(str(path))

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ------------------------------------------------------------------------------
# Startup / Shutdown Events
# ------------------------------------------------------------------------------

# async def _verify_disabled_accounts_loop():
#     """后台验证禁用账号任务"""
#     while True:
#         try:
#             await asyncio.sleep(1800)
#             async with _conn() as conn:
#                 accounts = await _list_disabled_accounts(conn)
#                 if accounts:
#                     for account in accounts:
#                         other = account.get('other')
#                         if other:
#                             try:
#                                 other_dict = json.loads(other) if isinstance(other, str) else other
#                                 if other_dict.get('failedReason') == 'AccessDenied':
#                                     continue
#                             except:
#                                 pass
#                         try:
#                             verify_success, fail_reason = await verify_account(account)
#                             now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#                             if verify_success:
#                                 await conn.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, account['id']))
#                             elif fail_reason:
#                                 other_dict = {}
#                                 if account.get('other'):
#                                     try:
#                                         other_dict = json.loads(account['other']) if isinstance(account['other'], str) else account['other']
#                                     except:
#                                         pass
#                                 other_dict['failedReason'] = fail_reason
#                                 await conn.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, account['id']))
#                             await conn.commit()
#                         except Exception:
#                             pass
#         except Exception:
#             pass

@app.on_event("startup")
async def startup_event():
    """Initialize database and start background tasks on startup."""
    await _init_global_client()
    await _ensure_db()
    asyncio.create_task(_refresh_stale_tokens())
    global _stats_worker_task
    _stats_worker_task = asyncio.create_task(_stats_worker())
    # asyncio.create_task(_verify_disabled_accounts_loop())

@app.on_event("shutdown")
async def shutdown_event():
    global _stats_worker_task
    if _stats_worker_task:
        # Flush pending updates before shutting down.
        await _stats_queue.join()
        _stats_worker_task.cancel()
        try:
            await _stats_worker_task
        except asyncio.CancelledError:
            pass
    await _close_global_client()
    await close_db()

# ------------------------------------------------------------------------------
# Batch Import Routes
# ------------------------------------------------------------------------------

class BatchImportRequest(BaseModel):
    accountData: str
    dryRun: bool = False

@app.get("/batch-import", response_class=HTMLResponse)
async def batch_import_page(request: Request):
    """批量导入页面"""
    return templates.TemplateResponse("batch_import.html", {"request": request})

@app.post("/api/batch-import")
async def batch_import_api(req: BatchImportRequest):
    """批量导入API"""
    try:
        lines = [line.strip() for line in req.accountData.strip().split('\n') if line.strip()]

        if not lines:
            return JSONResponse(
                status_code=400,
                content={"error": "没有找到有效的账户数据"}
            )

        accounts = []
        errors = []

        # 解析所有账户
        for i, line in enumerate(lines, 1):
            try:
                parts = line.strip().split('|')
                if len(parts) < 6:
                    raise ValueError(f"格式错误: 需要至少6个字段，实际{len(parts)}个")

                email = parts[0]
                password = parts[1]
                client_id = parts[2]
                client_secret = parts[3]
                refresh_token = parts[4]
                access_token = parts[5]

                # 生成唯一ID
                account_id = str(uuid.uuid4())
                label = email.split('@')[0] if '@' in email else email

                account = {
                    'id': account_id,
                    'label': label,
                    'clientId': client_id,
                    'clientSecret': client_secret,
                    'refreshToken': refresh_token,
                    'accessToken': access_token,
                    'other': json.dumps({
                        'email': email,
                        'password': password,
                        'imported_at': datetime.now().isoformat()
                    }),
                    'last_refresh_time': datetime.now().isoformat(),
                    'last_refresh_status': 'success',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'enabled': 1,
                    'error_count': 0,
                    'success_count': 0
                }
                accounts.append(account)

            except Exception as e:
                errors.append(f"账户 {i}: {str(e)}")

        if req.dryRun:
            # 预览模式
            message = f"✅ 预览模式结果:\n"
            message += f"📊 总计: {len(lines)} 行数据\n"
            message += f"✅ 成功解析: {len(accounts)} 个账户\n"
            message += f"❌ 解析错误: {len(errors)} 个账户\n\n"

            if accounts:
                message += "📋 成功解析的账户:\n"
                for account in accounts[:5]:  # 只显示前5个
                    message += f"  - {account['label']} (ID: {account['id'][:8]}...)\n"
                if len(accounts) > 5:
                    message += f"  ... 还有 {len(accounts) - 5} 个账户\n"

            if errors:
                message += "\n❌ 解析错误:\n"
                for error in errors[:5]:  # 只显示前5个错误
                    message += f"  - {error}\n"
                if len(errors) > 5:
                    message += f"  ... 还有 {len(errors) - 5} 个错误\n"

            return {"success": True, "message": message}

        # 实际导入
        if not accounts:
            return JSONResponse(
                status_code=400,
                content={"error": "没有成功解析的账户"}
            )

        # 使用数据库后端系统
        db = get_database_backend()

        imported_count = 0
        duplicate_count = 0

        for account in accounts:
            try:
                # 检查是否已存在相同的clientId
                existing = await db.fetchone(
                    "SELECT id FROM accounts WHERE clientId = ?",
                    (account['clientId'],)
                )

                if existing:
                    duplicate_count += 1
                    continue

                # 插入新账户
                await db.execute("""
                    INSERT INTO accounts (
                        id, label, clientId, clientSecret, refreshToken, accessToken,
                        other, last_refresh_time, last_refresh_status, created_at,
                        updated_at, enabled, error_count, success_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    account['id'], account['label'], account['clientId'],
                    account['clientSecret'], account['refreshToken'], account['accessToken'],
                    account['other'], account['last_refresh_time'], account['last_refresh_status'],
                    account['created_at'], account['updated_at'], account['enabled'],
                    account['error_count'], account['success_count']
                ))
                imported_count += 1

            except Exception as e:
                errors.append(f"导入账户 {account['label']} 失败: {str(e)}")

        message = f"✅ 导入完成!\n"
        message += f"📊 总计: {len(lines)} 行数据\n"
        message += f"✅ 成功导入: {imported_count} 个账户\n"
        message += f"⏭️  跳过重复: {duplicate_count} 个账户\n"
        message += f"❌ 导入失败: {len(errors)} 个账户"

        if errors:
            message += "\n\n❌ 错误详情:\n"
            for error in errors[:5]:
                message += f"  - {error}\n"
            if len(errors) > 5:
                message += f"  ... 还有 {len(errors) - 5} 个错误\n"

        return {"success": True, "message": message}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"批量导入失败: {str(e)}"}
        )

