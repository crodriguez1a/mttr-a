"""
MTTR-A web dashboard — FastAPI backend.

Security model
--------------
API keys arrive in the POST body only (never query params or headers that
land in access logs).  The key is used once to instantiate the provider
client, then explicitly overwritten and deleted before the background thread
starts.  It is never stored in the session dict, never written to disk, and
never included in any response or log line.

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

from mttr_a import BenchmarkConfig, MockProvider, ProductionRunner, ProviderConfig, ProviderKind
from mttr_a.providers import embedding_confidence
from mttr_a.sinks import TelemetrySink
from mttr_a_simulation import Episode

_HTML = (Path(__file__).parent / "static" / "index.html").read_text()

SESSION_TTL = timedelta(minutes=15)


# ── Session store ──────────────────────────────────────────────────────────────

@dataclass
class _Session:
    id: str
    queue: Queue = field(default_factory=Queue)
    created_at: datetime = field(default_factory=datetime.utcnow)
    done: bool = False


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


# ── Streaming sink ─────────────────────────────────────────────────────────────

class _StreamingSink:
    def __init__(self, q: Queue, n_runs: int) -> None:
        self._q = q
        self._n = n_runs

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        self._q.put({
            "type": "episode",
            "run_id": episode.run_id,
            "total": self._n,
            "confidence": round(episode.confidence, 4),
            "drift": episode.drift_detected,
            "delta_t": round(episode.delta_t, 3) if episode.drift_detected else None,
            "reflex": episode.reflex_mode,
        })

    def flush(self) -> None:
        pass


# ── Provider factory ───────────────────────────────────────────────────────────

def _build_provider(kind: str, cfg: ProviderConfig, api_key: str):
    if kind == "mock":
        return MockProvider(cfg)

    if kind == "azure_openai":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Azure provider requires langchain-openai: pip install langchain-openai"
            ) from exc

        from mttr_a.providers import BaseLLMProvider, LLMResponse
        import time as _t

        _llm = AzureChatOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            azure_deployment=cfg.azure_deployment,
            api_version=cfg.azure_api_version,
            api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        class _AzureProvider(BaseLLMProvider):
            def invoke(self, query: str, context: str = "") -> LLMResponse:
                from langchain_core.messages import HumanMessage
                t0 = _t.perf_counter()
                answer = _llm.invoke([HumanMessage(content=query)])
                return LLMResponse(
                    content=answer.content,
                    confidence=embedding_confidence(query, answer.content),
                    latency_s=_t.perf_counter() - t0,
                )

        return _AzureProvider()

    if kind == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude provider requires langchain-anthropic: pip install langchain-anthropic"
            ) from exc

        from mttr_a.providers import BaseLLMProvider, LLMResponse
        import time as _t

        _llm = ChatAnthropic(
            model=cfg.model_id,
            api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        class _ClaudeProvider(BaseLLMProvider):
            def invoke(self, query: str, context: str = "") -> LLMResponse:
                from langchain_core.messages import HumanMessage
                t0 = _t.perf_counter()
                answer = _llm.invoke([HumanMessage(content=query)])
                return LLMResponse(
                    content=answer.content,
                    confidence=embedding_confidence(query, answer.content),
                    latency_s=_t.perf_counter() - t0,
                )

        return _ClaudeProvider()

    if kind == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Google provider requires langchain-google-genai: pip install langchain-google-genai"
            ) from exc

        from mttr_a.providers import BaseLLMProvider, LLMResponse
        import time as _t

        _llm = ChatGoogleGenerativeAI(
            model=cfg.model_id,
            google_api_key=api_key,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
        )

        class _GoogleProvider(BaseLLMProvider):
            def invoke(self, query: str, context: str = "") -> LLMResponse:
                from langchain_core.messages import HumanMessage
                t0 = _t.perf_counter()
                answer = _llm.invoke([HumanMessage(content=query)])
                return LLMResponse(
                    content=answer.content,
                    confidence=embedding_confidence(query, answer.content),
                    latency_s=_t.perf_counter() - t0,
                )

        return _GoogleProvider()

    raise ValueError(f"Unsupported provider: {kind}")


# ── Background runner ──────────────────────────────────────────────────────────

def _run_in_thread(session: _Session, cfg: BenchmarkConfig, provider, query_pool=None) -> None:
    try:
        sink = _StreamingSink(session.queue, cfg.n_runs)
        metrics = ProductionRunner(cfg, provider, sink, query_pool=query_pool).run()
        session.queue.put({
            "type": "complete",
            "metrics": {
                "nrr":        round(metrics.nrr_sys, 4),
                "pi_up":      round(metrics.pi_up_sys, 4),
                "med_ttr_a":  round(metrics.med_ttr_a_sys, 3),
                "mtbf":       round(metrics.mtbf_sys, 3),
                "drift_rate": round(metrics.drift_rate, 4),
                "p90":        round(metrics.p90_sys, 3),
                "nrr_alpha":  round(metrics.nrr_alpha, 4),
                "n_runs":     metrics.n_runs,
                "drift_events": metrics.drift_events,
            },
        })
    except Exception as exc:
        session.queue.put({"type": "error", "message": str(exc)})
    finally:
        session.done = True


# ── Request model ──────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    provider: str = "mock"
    n_runs: int = 30
    seed: int = 42
    simulate_latency: bool = True
    azure_endpoint: str = ""
    azure_deployment: str = ""
    claude_model: str = "claude-sonnet-4-6"
    google_model: str = "gemini-2.0-flash"
    api_key: str = ""
    custom_prompt: str = ""  # if set, used for every episode instead of the built-in pool


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MTTR-A", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.post("/api/run")
async def start_run(req: RunRequest):
    kind_map = {"mock": ProviderKind.MOCK, "azure_openai": ProviderKind.AZURE_OPENAI}
    kind_enum = kind_map.get(req.provider, ProviderKind.MOCK)
    model_id = (
        req.claude_model if req.provider == "claude" else
        req.google_model if req.provider == "google" else
        "mock-model"
    )
    provider_cfg = ProviderConfig(
        kind=kind_enum,
        model_id=model_id,
        mock_simulate_latency=req.simulate_latency,
        azure_endpoint=req.azure_endpoint,
        azure_deployment=req.azure_deployment,
        azure_api_version="2024-02-01",
    )
    cfg = BenchmarkConfig(
        provider=provider_cfg,
        n_runs=max(5, min(req.n_runs, 200)),
        seed=req.seed,
        verbose=False,
    )

    _key = req.api_key
    try:
        provider = _build_provider(req.provider, provider_cfg, _key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        _key = "\x00" * len(_key)
        del _key

    query_pool = [req.custom_prompt.strip()] if req.custom_prompt.strip() else None

    session = _new_session()
    Thread(target=_run_in_thread, args=(session, cfg, provider, query_pool), daemon=True).start()
    return {"session_id": session.id}


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
