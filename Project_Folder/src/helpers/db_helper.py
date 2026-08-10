"""Database Migration and Initialization Helper using Alembic."""

import os
from pathlib import Path
from alembic import command
from alembic.config import Config

from src.core.logger import get_logger

logger = get_logger(__name__)


def run_alembic_migrations() -> bool:
    """Programmatically run Alembic migrations up to head revision."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        ini_path = project_root / "alembic.ini"

        if not ini_path.exists():
            logger.warning(f"alembic.ini not found at {ini_path}. Skipping automatic migrations.")
            return False

        logger.info("▶ Running Alembic database migrations (upgrade head)...")
        alembic_cfg = Config(str(ini_path))
        # Ensure alembic script location points to absolute directory
        alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))

        command.upgrade(alembic_cfg, "head")
        logger.info("✔ Alembic database migrations completed successfully")
        return True

    except Exception as err:
        logger.error(f"Alembic migration failed: {err}")
        return False
