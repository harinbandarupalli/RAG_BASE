"""Step 12: Bidirectional WebSocket Streaming Route with Client Kill Switch.

Handles real-time streaming of lifecycle events:
- 'status': Updates on current pipeline step (guardrail, decomposition, hybrid search, rerank)
- 'token': Progressive token stream
- 'citations': Structured source attribution drawer
- 'telemetry': Granular waterfall latency, token counts, and USD cost breakdown
- 'cancelled': Graceful cancellation when client fires kill switch
"""

import asyncio
import json
from typing import Any
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.graph import execute_rag_pipeline
from src.core.logger import get_logger
from src.helpers.task_supervisor import TaskSupervisor

logger = get_logger(__name__)
router = APIRouter(tags=["WebSocket Chat Stream"])


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket) -> None:
    """Bidirectional WebSocket endpoint for conversational RAG with process cancellation."""
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/chat")

    current_task: asyncio.Task[Any] | None = None
    task_id: str = f"task_{uuid.uuid4().hex[:8]}"

    async def run_pipeline_and_stream(query_payload: dict[str, Any]) -> None:
        """Run the 12-step pipeline and stream progressive events."""
        user_query = query_payload.get("user_query", "")
        session_id = query_payload.get("session_id", f"sess_{uuid.uuid4().hex[:8]}")
        provider = query_payload.get("model_provider")
        model = query_payload.get("model_name")
        temp = query_payload.get("temperature", 0.2)

        try:
            # Emitting status updates
            await websocket.send_json({"event": "status", "data": "Validating input guardrails & security..."})
            await asyncio.sleep(0.05)

            await websocket.send_json({"event": "status", "data": "Checking Redis semantic cache..."})
            await asyncio.sleep(0.05)

            await websocket.send_json({"event": "status", "data": "Decomposing query & running parallel hybrid retrieval..."})
            await asyncio.sleep(0.05)

            await websocket.send_json({"event": "status", "data": "Executing cross-encoder reranker & confidence gate..."})
            await asyncio.sleep(0.05)

            await websocket.send_json({"event": "status", "data": "Processing with LLM & strict remote prompt hub..."})

            # Execute the compiled LangGraph pipeline
            final_state = await execute_rag_pipeline(
                user_query=user_query,
                session_id=session_id,
                model_provider=provider,
                model_name=model,
                temperature=temp,
            )

            # Stream response text chunks
            response_text = final_state.get("llm_response", "")
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await websocket.send_json({"event": "token", "data": chunk})
                await asyncio.sleep(0.015)  # Natural progressive streaming pace

            # Emit citations payload
            citations = final_state.get("citations", [])
            await websocket.send_json({"event": "citations", "data": citations})

            # Emit waterfall telemetry breakdown
            telemetry = final_state.get("telemetry", {})
            await websocket.send_json({"event": "telemetry", "data": telemetry})

            # Emit completion event
            await websocket.send_json({"event": "done", "trace_id": final_state.get("trace_id")})

        except asyncio.CancelledError:
            logger.warning(f"Task '{task_id}' was cancelled via client kill switch")
            await websocket.send_json({"event": "cancelled", "message": "Generation halted by user."})
            raise
        except Exception as err:
            logger.error(f"WebSocket execution error: {err}")
            await websocket.send_json({"event": "error", "message": str(err)})

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            action = data.get("action")

            # 1. Action: Cancel / Kill Generation
            if action in ("kill", "cancel"):
                if current_task and not current_task.done():
                    TaskSupervisor.cancel_task(task_id)
                continue

            # 2. Action: Query Execution
            elif action == "query":
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                current_task = asyncio.create_task(run_pipeline_and_stream(data))
                TaskSupervisor.register_task(task_id, current_task)

                def on_done(t: asyncio.Task[Any]) -> None:
                    TaskSupervisor.unregister_task(task_id)

                current_task.add_done_callback(on_done)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        if current_task and not current_task.done():
            TaskSupervisor.cancel_task(task_id)
    except Exception as err:
        logger.error(f"WebSocket connection error: {err}")
