"""Task Supervisor, Active Process Watchdog & Kill Switch.

Provides:
- Active execution task registry
- Bidirectional kill-switch: cancel_task() terminates running async tasks immediately
- Watchdogs for token consumption budget and execution timeouts
"""

import asyncio
from typing import Any

from src.core.config import get_settings
from src.core.exceptions import BudgetExceededError, TaskCancelledError, TaskTimeoutError
from src.core.logger import get_logger

logger = get_logger(__name__)


class TaskSupervisor:
    """Supervisor for managing running pipeline tasks and kill switches."""

    _active_tasks: dict[str, asyncio.Task[Any]] = {}

    @classmethod
    def register_task(cls, task_id: str, task: asyncio.Task[Any]) -> None:
        """Register an active running asyncio task."""
        cls._active_tasks[task_id] = task
        logger.debug(f"Registered active task '{task_id}' (Total active: {len(cls._active_tasks)})")

    @classmethod
    def unregister_task(cls, task_id: str) -> None:
        """Remove task from registry upon completion."""
        cls._active_tasks.pop(task_id, None)
        logger.debug(f"Unregistered task '{task_id}' (Total active: {len(cls._active_tasks)})")

    @classmethod
    def cancel_task(cls, task_id: str) -> bool:
        """Immediately terminate an active task via kill switch."""
        task = cls._active_tasks.get(task_id)
        if task and not task.done():
            logger.warning(f"🛑 KILL SWITCH ACTIVATED: Cancelling task '{task_id}' immediately")
            task.cancel()
            cls.unregister_task(task_id)
            return True
        logger.debug(f"Task '{task_id}' not found or already completed")
        return False

    @classmethod
    def list_active_tasks(cls) -> list[str]:
        """Return list of currently active task IDs."""
        return [tid for tid, t in cls._active_tasks.items() if not t.done()]

    @classmethod
    def verify_token_budget(cls, current_tokens: int) -> None:
        """Raise BudgetExceededError if token limit is exceeded."""
        settings = get_settings()
        limit = settings.supervisor.max_tokens_budget_per_run
        if current_tokens > limit:
            raise BudgetExceededError(
                f"Generation exceeded token budget: {current_tokens} > {limit}. Process halted."
            )
