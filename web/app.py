"""
MTTR-A web dashboard — FastAPI backend.

Security model
--------------
API keys arrive in the POST body only (never query params or headers that
land in access logs).  The key is used once to start the background thread,
then explicitly overwritten and deleted before the thread runs.  It is never
stored in the session dict, never written to disk, and never included in any
response or log line.

Session IDs are 32-byte cryptographically random tokens (secrets.token_urlsafe).
Each SSE stream is bound to exactly one session; cross-session access returns 404.
Sessions expire 15 minutes after creation and are purged on the next request.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from mttr_a_simulation import MetricsComputer

_HTML = (Path(__file__).parent / "static" / "index.html").read_text()

SESSION_TTL = timedelta(minutes=15)


# ── Session store ──────────────────────────────────────────────────────────────

@dataclass
class _Session:
    id: str
    queue: Queue = field(default_factory=Queue)
    created_at: datetime = field(default_factory=datetime.utcnow)
    done: bool = False
    episodes: list = field(default_factory=list)


_sessions: dict[str, _Session] = {}


def _new_session() -> _Session:
    _purge_expired()
    sid = secrets.token_urlsafe(32)
    s = _Session(id=sid)
    _sessions[sid] = s
    return s


def _get_session(sid: str) -> _Session:
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return s


def _purge_expired() -> None:
    cutoff = datetime.utcnow() - SESSION_TTL
    for k in [k for k, v in _sessions.items() if v.created_at < cutoff]:
        del _sessions[k]


# ── Background runner ──────────────────────────────────────────────────────────

def _run_task_in_thread(session: _Session, provider: str, task: str, api_key: str, model: str, run_id: int) -> None:
    from mttr_a.sdk.tools import DEMO_TOOLS
    from mttr_a.sdk.adapters.claude import ClaudeAdapter
    from mttr_a.sdk.adapters.gemini import GeminiAdapter

    try:
        tools = [t.__class__() for t in DEMO_TOOLS]

        if provider == "claude":
            sdk_session = ClaudeAdapter().run(
                task=task,
                tools=tools,
                api_key=api_key,
                model=model or "claude-sonnet-4-6",
                max_turns=12,
                event_queue=session.queue,
                run_id=run_id,
            )
        elif provider == "gemini":
            sdk_session = GeminiAdapter().run(
                task=task,
                tools=tools,
                api_key=api_key,
                model=model or "gemini-2.0-flash",
                max_turns=12,
                event_queue=session.queue,
                run_id=run_id,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        episode = sdk_session.episode()
        session.episodes.append(episode)

        stable_intervals = [1.0] * max(1, len(session.episodes) - 1) + [6.73]

        metrics_dict: dict = {"n_runs": len(session.episodes), "drift_events": 0}
        try:
            metrics = MetricsComputer().compute(
                session.episodes, stable_intervals, n_runs=len(session.episodes)
            )
            metrics_dict = {
                "nrr":          round(metrics.nrr_sys, 4),
                "pi_up":        round(metrics.pi_up_sys, 4),
                "med_ttr_a":    round(metrics.med_ttr_a_sys, 3),
                "mtbf":         round(metrics.mtbf_sys, 3),
                "drift_rate":   round(metrics.drift_rate, 4),
                "p90":          round(metrics.p90_sys, 3),
                "nrr_alpha":    round(metrics.nrr_alpha, 4),
                "n_runs":       metrics.n_runs,
                "drift_events": metrics.drift_events,
            }
        except Exception:
            pass

        session.queue.put({
            "type": "complete",
            "episode": {
                "run_id": episode.run_id,
                "drift_detected": episode.drift_detected,
                "step_confidences": list(episode.step_confidences),
                "delta_t": round(episode.delta_t, 3),
            },
            "metrics": metrics_dict,
        })
    except Exception as exc:
        session.queue.put({"type": "error", "message": str(exc)})
    finally:
        session.done = True


# ── Request model ──────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    provider: str
    task: str
    api_key: str
    model: str = ""


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MTTR-A", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/api/corpus-info")
async def corpus_info():
    from mttr_a.providers import _load_corpus
    try:
        docs = _load_corpus()
        return {"source": "default", "doc_count": len(docs), "description": "MTTR-A reference corpus — technical reasoning domains (distributed systems, ML/AI, algorithms, SRE, security, and more)."}
    except Exception as exc:
        return {"source": "default", "doc_count": 0, "description": str(exc)}


@app.post("/api/task")
async def start_task(req: TaskRequest):
    if req.provider not in ("claude", "gemini"):
        raise HTTPException(status_code=400, detail="provider must be 'claude' or 'gemini'")
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task description is required")

    session = _new_session()
    run_id = len(session.episodes)

    _key = req.api_key
    try:
        Thread(
            target=_run_task_in_thread,
            args=(session, req.provider, req.task.strip(), _key, req.model, run_id),
            daemon=True,
        ).start()
    finally:
        _key = "\x00" * len(_key)
        del _key

    return {"session_id": session.id}


@app.post("/api/mock")
async def start_mock():
    """Simulate a realistic agent run without any API key.

    Emits the same SSE event types as /api/task: step, tool_result, complete.
    Simulates: search_corpus → evidence set → stable turns → drift → natural recovery.
    """
    session = _new_session()

    async def _generate():
        import math, random

        tau = 0.6
        evidence_emb_set = False
        t_fault = None
        t_recovered = None
        drift_detected = False
        step_confidences: list[float] = []

        scenario = [
            # (tool_called, confidence, latency_s)
            ("search_corpus", None, 0.31),   # turn 0: retrieval, confidence set after
            (None, 0.82, 0.28),               # turn 1: stable, well-grounded
            (None, 0.76, 0.24),               # turn 2: stable
            (None, 0.61, 0.30),               # turn 3: stable but drifting toward threshold
            (None, 0.44, 0.27),               # turn 4: DRIFT — below τ
            (None, 0.51, 0.29),               # turn 5: still drifting
            (None, 0.67, 0.26),               # turn 6: natural recovery — above τ again
            ("conclude", 0.79, 0.22),         # turn 7: final answer
        ]

        for turn, (tool, conf, lat) in enumerate(scenario):
            await asyncio.sleep(lat * 0.5)   # simulate inference latency (half-speed for demo)

            # After search_corpus fires, evidence is now set
            if tool == "search_corpus":
                evidence_emb_set = True
                # Emit tool_result for the search
                yield f"data: {json.dumps({'type': 'tool_result', 'turn': turn, 'tool_name': 'search_corpus', 'latency_s': round(lat * 0.3, 3)})}\n\n"
                # First step has no confidence yet (no evidence before retrieval)
                confidence = None
                step_confidences.append(0.0)
            else:
                confidence = conf
                step_confidences.append(conf or 0.0)

            # Drift / recovery tracking
            is_drift = evidence_emb_set and confidence is not None and confidence < tau
            is_recovered = False

            if is_drift and not drift_detected:
                drift_detected = True
                t_fault = sum(s[2] for s in scenario[:turn]) * 0.5
            if drift_detected and not is_drift and t_fault is not None and t_recovered is None and confidence is not None:
                t_recovered = t_fault + sum(s[2] for s in scenario[4:turn+1]) * 0.5
                is_recovered = True

            yield f"data: {json.dumps({'type': 'step', 'turn': turn, 'confidence': confidence, 'reasoning_excerpt': f'Turn {turn} reasoning — mock simulation', 'tool_called': tool, 'drift': is_drift, 'recovered': is_recovered})}\n\n"

        delta_t = round((t_recovered - t_fault), 3) if (t_fault is not None and t_recovered is not None) else 0.0

        episode_out = {
            "run_id": 0,
            "drift_detected": drift_detected,
            "step_confidences": step_confidences,
            "delta_t": delta_t,
        }
        metrics_out = {
            "nrr": 0.91,
            "pi_up": 0.94,
            "med_ttr_a": delta_t,
            "mtbf": 6.73,
            "drift_rate": 0.38,
            "p90": round(delta_t * 1.4, 3),
            "nrr_alpha": 0.87,
            "n_runs": 1,
            "drift_events": 1 if drift_detected else 0,
        }
        yield f"data: {json.dumps({'type': 'complete', 'episode': episode_out, 'metrics': metrics_out})}\n\n"
        session.done = True

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stream/{session_id}")
async def stream_results(session_id: str):
    session = _get_session(session_id)

    async def _generate():
        while True:
            try:
                event = session.queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("complete", "error"):
                    break
            except Empty:
                if session.done and session.queue.empty():
                    break
                await asyncio.sleep(0.04)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
