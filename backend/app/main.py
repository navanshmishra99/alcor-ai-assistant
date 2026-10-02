from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .api.chat import router as chat_router
from .knowledge.ingest import ensure_knowledge_base_loaded


# =========================================================
# Logging
# =========================================================

logger = logging.getLogger("ask_alcor.startup")


# =========================================================
# Environment helpers
# =========================================================

def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable safely."""
    try:
        value = int(os.getenv(name, str(default)).strip())
        return value
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer for %s. Using default=%s.",
            name,
            default,
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable safely."""
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning(
        "Invalid boolean for %s. Using default=%s.",
        name,
        default,
    )

    return default


def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated list from the environment."""
    value = os.getenv(name, "").strip()

    if not value:
        return default

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


# =========================================================
# Application configuration
# =========================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
).strip()


# Ollama duration values need a unit.
#
# Examples:
#   30s
#   5m
#   1h
#   30m
#
# This prevents invalid values such as "-1" from being sent
# to Ollama. The actual value remains configurable through .env.
_DURATION_PATTERN = re.compile(
    r"^\d+(?:\.\d+)?(?:ns|us|µs|ms|s|m|h)$",
    re.IGNORECASE,
)


def _env_duration(name: str, default: str) -> str:
    """Read and validate an Ollama-style duration."""
    value = os.getenv(name, default).strip()

    if _DURATION_PATTERN.fullmatch(value):
        return value

    logger.warning(
        "Invalid duration for %s=%r. Using default=%s.",
        name,
        value,
        default,
    )

    return default


OLLAMA_KEEP_ALIVE = _env_duration(
    "OLLAMA_KEEP_ALIVE",
    "30m",
)

MODEL_WARMUP_ENABLED = _env_bool(
    "MODEL_WARMUP_ENABLED",
    True,
)

MODEL_WARMUP_TIMEOUT = _env_int(
    "MODEL_WARMUP_TIMEOUT",
    180,
)

KB_LOAD_TIMEOUT = _env_int(
    "KB_LOAD_TIMEOUT",
    300,
)


# =========================================================
# CORS configuration
# =========================================================

DEFAULT_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

ALLOWED_ORIGINS = _env_list(
    "ALLOWED_ORIGINS",
    DEFAULT_ORIGINS,
)


# =========================================================
# Knowledge-base state
# =========================================================

kb_status = {
    "ready": False,
    "status": "loading",
    "document_count": 0,
    "chunk_count": 0,
}

model_status = {
    "ready": False,
    "status": "loading",
}


# =========================================================
# Knowledge-base loading
# =========================================================

def load_knowledge_base() -> None:
    """
    Load or validate the knowledge base in the background.

    The API is considered ready only after the KB loader
    successfully returns a state.
    """

    kb_status.update(
        ready=False,
        status="loading",
    )

    try:
        state = ensure_knowledge_base_loaded()

        kb_status.update(
            ready=True,
            status=state.get("status", "ready"),
            document_count=state.get("document_count", 0),
            chunk_count=state.get("chunk_count", 0),
        )

        logger.info(
            "Knowledge base ready: status=%s, documents=%s, chunks=%s",
            kb_status["status"],
            kb_status["document_count"],
            kb_status["chunk_count"],
        )

    except Exception:
        kb_status.update(
            ready=False,
            status="failed",
        )

        logger.exception(
            "Knowledge base bootstrap failed."
        )


# =========================================================
# Model warm-up
# =========================================================

def warm_up_model() -> None:
    """
    Ask Ollama to load the configured model into memory.

    This is best-effort only. A warm-up failure does not make
    the application itself fail because the actual AI request
    can still attempt to load the model later.
    """

    if not MODEL_WARMUP_ENABLED:
        model_status.update(
            ready=True,
            status="disabled",
        )
        logger.info("Model warm-up disabled.")
        return

    try:
        with httpx.Client(
            trust_env=False,
            timeout=MODEL_WARMUP_TIMEOUT,
        ) as client:

            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
            )

            if response.is_success:
                model_status.update(
                    ready=True,
                    status="ready",
                )
                logger.info(
                    "Language model warm-up completed: model=%s keep_alive=%s",
                    OLLAMA_MODEL,
                    OLLAMA_KEEP_ALIVE,
                )
            else:
                model_status.update(
                    ready=False,
                    status=f"http_{response.status_code}",
                )
                logger.warning(
                    "Language model warm-up returned HTTP %s.",
                    response.status_code,
                )

    except Exception:
        logger.warning(
            "Language model warm-up skipped.",
            exc_info=True,
        )


# =========================================================
# Application lifespan
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start background initialization without blocking the web server.
    """

    threading.Thread(
        target=load_knowledge_base,
        name="kb-loader",
        daemon=True,
    ).start()

    threading.Thread(
        target=warm_up_model,
        name="model-warmup",
        daemon=True,
    ).start()

    yield


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title=os.getenv(
        "APP_TITLE",
        "Alcor AI Assistant",
    ),
    lifespan=lifespan,
)


# =========================================================
# Readiness middleware
# =========================================================

@app.middleware("http")
async def wait_for_knowledge_base(
    request: Request,
    call_next,
):
    """
    Prevent chat requests from reaching retrieval while the
    knowledge base is still initializing.

    The widget can retry the 503 response using Retry-After.
    """

    is_chat_request = (
        request.method.upper() == "POST"
        and request.url.path.rstrip("/") == "/api/chat"
    )

    if is_chat_request and (
        not kb_status["ready"] or
        not model_status["ready"]
    ):

        if kb_status["status"] == "failed" or model_status["status"] == "failed":
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Knowledge base initialization failed."
                },
                headers={
                    "Retry-After": "5",
                },
            )

        return JSONResponse(
            status_code=503,
            content={
                "detail": "Service is starting up."
            },
            headers={
                "Retry-After": "3",
            },
        )

    return await call_next(request)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Routes
# =========================================================

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <head>
        <title>AI Assistant</title>
        <meta charset="utf-8" />

        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px auto;
            max-width: 900px;
            color: #1f2937;
            background: #f8fafc;
            line-height: 1.6;
          }

          .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 32px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
          }

          h1 {
            margin-top: 0;
            font-size: 2.2rem;
          }

          .status {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            border-radius: 999px;
            padding: 6px 12px;
            font-weight: 700;
            margin-bottom: 14px;
          }

          a {
            color: #2563eb;
            text-decoration: none;
          }

          a:hover {
            text-decoration: underline;
          }
        </style>
      </head>

      <body>
        <div class="card">
          <div class="status">API Online</div>

          <h1>AI Assistant</h1>

          <p>
            The backend is running successfully.
          </p>

          <p>
            <a href="/docs">Open API docs</a>
            |
            <a href="/health">Health check</a>
          </p>

          <p>
            Use the Swagger UI to test the assistant and review
            request/response schemas.
          </p>
        </div>
      </body>
    </html>
    """


@app.get("/health")
def health_check():
    """
    Return application and knowledge-base readiness information.
    """

    return {
        "status": "ok",
        "knowledge_base": kb_status,
        "model": {
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "warmup_enabled": MODEL_WARMUP_ENABLED,
            "ready": model_status["ready"],
            "status": model_status["status"],
        },
    }


# =========================================================
# API routers
# =========================================================

app.include_router(
    chat_router,
    prefix="/api",
)