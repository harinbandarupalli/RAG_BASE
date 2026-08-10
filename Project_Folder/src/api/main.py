"""FastAPI Application Entrypoint, Lifespan Management, and Global Handlers."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.databases.postgres import PostgresManager
from src.adapters.databases.redis_cache import RedisCacheAdapter
from src.api.routes.chat_ws import router as ws_router
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.models import router as models_router
from src.core.config import get_settings
from src.core.exceptions import PromptHubError, PromptNotFoundError, RAGBaseException
from src.core.logger import get_logger
from src.observability.langsmith_adapter import LangSmithAdapter
from src.observability.phoenix_adapter import PhoenixAdapter

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize database schemas, vector extensions, and observability."""
    settings = get_settings()
    logger.info(f"▶ Starting {settings.app.name} (env: {settings.app.environment})...")

    # 1. Setup Observability
    LangSmithAdapter.setup_environment()
    PhoenixAdapter.initialize()

    # 2. Run Alembic database migrations & initialize pgvector
    try:
        from src.helpers.db_helper import run_alembic_migrations
        run_alembic_migrations()
        await PostgresManager.initialize_schema()
    except Exception as err:
        logger.warning(f"Could not auto-run database migrations on startup: {err}")

    yield

    # 3. Shutdown cleanup
    logger.info("Shutting down database connection pools...")
    await PostgresManager.close()
    await RedisCacheAdapter.close()
    logger.info("✔ Cleanup completed")


def create_application() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app.name,
        version="0.1.0",
        description="Production-grade, Hexagonal LangChain & LangGraph RAG Starter Kit",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(PromptNotFoundError)
    async def prompt_not_found_handler(request: Request, exc: PromptNotFoundError) -> JSONResponse:
        logger.error(f"PromptHub Error: {exc.message}")
        return JSONResponse(
            status_code=502,
            content={
                "error": "PromptNotFoundError",
                "message": exc.message,
                "policy": "Strict governance: NO local fallback permitted.",
            },
        )

    @app.exception_handler(PromptHubError)
    async def prompt_hub_handler(request: Request, exc: PromptHubError) -> JSONResponse:
        logger.error(f"PromptHub Unreachable: {exc.message}")
        return JSONResponse(
            status_code=502,
            content={
                "error": "PromptHubError",
                "message": exc.message,
                "policy": "Strict governance: NO local fallback permitted.",
            },
        )

    @app.exception_handler(RAGBaseException)
    async def domain_exception_handler(request: Request, exc: RAGBaseException) -> JSONResponse:
        logger.warning(f"Domain error: {exc.message}")
        return JSONResponse(status_code=400, content=exc.to_dict())

    # Include API Routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(ingest_router)
    app.include_router(ws_router)

    return app


app = create_application()
