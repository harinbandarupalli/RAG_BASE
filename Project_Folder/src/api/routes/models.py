"""Model Registry and Task Supervisor REST Endpoints."""

from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.helpers.model_switcher import ModelSwitcher
from src.helpers.task_supervisor import TaskSupervisor

router = APIRouter(prefix="/api/v1", tags=["Models & Supervisor"])


class SwitchModelRequest(BaseModel):
    provider: str
    model: str


@router.get("/models", summary="List Available and Active LLM Models")
async def list_models() -> dict[str, Any]:
    """Retrieve all available models by provider and currently active defaults."""
    return {
        "active_model": ModelSwitcher.get_active_model(),
        "available_models": ModelSwitcher.list_available_models(),
    }


@router.post("/models/active", summary="Dynamically Switch Active Model")
async def switch_model(payload: SwitchModelRequest) -> dict[str, Any]:
    """Set the globally active LLM provider and model."""
    updated = ModelSwitcher.set_active_model(payload.provider, payload.model)
    return {"message": "Active model updated successfully", "active_model": updated}


@router.post("/tasks/{task_id}/kill", summary="Cancel/Kill Running Task")
async def kill_task(task_id: str) -> dict[str, Any]:
    """Cancel a runaway or expensive task immediately via process kill switch."""
    success = TaskSupervisor.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already finished.")
    return {"status": "cancelled", "task_id": task_id, "message": "Task killed successfully."}
