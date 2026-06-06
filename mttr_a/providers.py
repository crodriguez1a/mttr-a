"""
LLM provider abstraction layer.

Each provider implements BaseLLMProvider and returns an LLMResponse that
includes a confidence score the drift-detection node uses directly.
Swapping providers is a one-line config change — the graph nodes are provider-agnostic.

Confidence strategy
-------------------
  Mock     — sampled from N(0.65, 0.15) + N(0.0, 0.05), clamped to [0, 1].
             This Gaussian simulates the distribution of retrieval-confidence
             scores (cos(query, top_doc)) that a real pipeline would produce —
             no corpus or embedding call is needed for mock runs.
             Parameters are tunable in data/mock_config/config.json.

  Real     — retrieval_confidence(query): the paper's exact signal.
             Embeds the query with all-MiniLM-L6-v2, finds the top-matching
             document in data/corpus/documents.txt, and returns the cosine
             similarity cos(query_emb, top_doc_emb).

             Corpus embeddings are pre-computed and L2-normalised on first use,
             so retrieval is a single matrix–vector dot product after warm-up.
             Requires: pip install sentence-transformers
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import ProviderConfig, ProviderKind

_DATA_DIR = Path(__file__).parent.parent / "data"

def _load_mock_cfg() -> dict:
    return json.loads((_DATA_DIR / "mock_config" / "config.json").read_text())

_MOCK_CFG = _load_mock_cfg()
_MOCK_CONF = _MOCK_CFG["confidence_distribution"]
_MOCK_LATENCY: dict[str, tuple[float, float]] = {
    k: (v[0], v[1])
    for k, v in _MOCK_CFG["mock_provider_latency"].items()
    if not k.startswith("_")
}
_TTFT_FRACTION: float = _MOCK_CFG["ttft_fraction"]["value"]

# ── Retrieval-based confidence (paper's exact signal) ─────────────────────────

_embed_model = None  # lazy-loaded, shared across all provider instances

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Retrieval confidence requires sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


_corpus_docs: list[str] | None = None
_corpus_embs = None  # shape (n_docs, dim), L2-normalized


def _load_corpus() -> list[str]:
    path = _DATA_DIR / "corpus" / "documents.txt"
    return [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _get_corpus_embs():
    global _corpus_docs, _corpus_embs
    if _corpus_embs is None:
        import numpy as np
        model = _get_embed_model()
        _corpus_docs = _load_corpus()
        embs = model.encode(_corpus_docs, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        _corpus_embs = embs / (norms + 1e-9)
    return _corpus_embs


def retrieval_confidence(query: str) -> float:
    """
    Cosine similarity between the query embedding and the top-matching document
    in data/corpus/documents.txt — exactly the paper's confidence signal.

    cos(query_emb, top_doc_emb): high when the query maps to a well-represented
    concept in the corpus, lower when it drifts to unfamiliar territory.
    Returns a value in [0, 1].

    Corpus embeddings are pre-computed and cached on first call.
    """
    import numpy as np
    model = _get_embed_model()
    q = model.encode([query], convert_to_numpy=True)
    q = q / (np.linalg.norm(q) + 1e-9)
    return float(max(0.0, min(1.0, (_get_corpus_embs() @ q.T).max())))


@dataclass(frozen=True)
class ToolCall:
    """
    Record of a single tool invocation made during a provider call.

    Populate this in real providers when the model calls external tools
    (search, SQL, code-exec, etc.) mid-generation.  Aggregated into
    tool_latency_s and n_tool_calls in the telemetry record.
    """

    name: str
    latency_s: float
    success: bool = True


@dataclass(frozen=True)
class LLMResponse:
    """Structured return type from any provider."""

    content: str
    confidence: float              # 0.0–1.0; used by check_drift_node
    latency_s: float               # wall-clock seconds for the full call

    # Gap 2: time-to-first-token — set when using a streaming provider;
    # None means full-completion timing is the only signal available.
    t_first_token: float | None = None

    # Gap 4: tool calls made during this generation.
    # Use a tuple (not list) to preserve frozen-dataclass immutability.
    tool_calls: tuple[ToolCall, ...] = ()


class BaseLLMProvider(ABC):
    """Common interface every provider must satisfy."""

    @abstractmethod
    def invoke(self, query: str, context: str = "") -> LLMResponse:
        """
        Call the underlying model and return a structured response.
        context is a hint to guide recovery reflexes (e.g. "retry", "rollback").
        """

    def health_check(self) -> bool:
        """Return True if the provider can be reached."""
        try:
            r = self.invoke("ping")
            return len(r.content) > 0
        except Exception:
            return False


# ── Mock provider ─────────────────────────────────────────────────────────────

class MockProvider(BaseLLMProvider):
    """
    Deterministic mock for local development and CI.
    Produces realistic confidence distributions and optional simulated latency
    so MTTR-A metrics look meaningful without real API calls.

    Swap to a real provider by changing ProviderConfig.kind — no other changes.
    """

    def __init__(self, config: ProviderConfig, seed: int = 42) -> None:
        self._simulate_latency = config.mock_simulate_latency
        self._rng = random.Random(seed)

    def invoke(self, query: str, context: str = "") -> LLMResponse:
        t0 = time.perf_counter()

        confidence = float(max(0.0, min(1.0,
            self._rng.gauss(mu=_MOCK_CONF["base_mu"], sigma=_MOCK_CONF["base_sigma"])
            + self._rng.gauss(mu=_MOCK_CONF["noise_mu"], sigma=_MOCK_CONF["noise_sigma"])
        )))

        if self._simulate_latency:
            lo, hi = _MOCK_LATENCY.get(context, _MOCK_LATENCY[""])
            time.sleep(self._rng.uniform(lo, hi))

        content = f"[mock] response to: {query[:60]}"
        elapsed = time.perf_counter() - t0
        return LLMResponse(
            content=content,
            confidence=confidence,
            latency_s=elapsed,
            t_first_token=t0 + elapsed * _TTFT_FRACTION,
        )


# ── AWS Bedrock ───────────────────────────────────────────────────────────────

class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock via langchain-aws.

    Authentication: standard boto3 credential chain
      (IAM role → env vars → ~/.aws/credentials)

    Required env vars / config:
      MTTR_MODEL_ID   e.g. anthropic.claude-3-5-sonnet-20241022-v2:0
      MTTR_AWS_REGION e.g. us-east-1

    Install: pip install langchain-aws
    """

    def __init__(self, config: ProviderConfig) -> None:
        try:
            from langchain_aws import ChatBedrock
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError as exc:
            raise ImportError(
                "Install langchain-aws: pip install langchain-aws"
            ) from exc

        self._llm = ChatBedrock(
            model_id=config.model_id,
            region_name=config.aws_region,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )
        self._HumanMessage = HumanMessage
        self._AIMessage = AIMessage

    def invoke(self, query: str, context: str = "") -> LLMResponse:
        from langchain_core.messages import HumanMessage
        t0 = time.perf_counter()
        answer = self._llm.invoke([HumanMessage(content=query)])
        latency = time.perf_counter() - t0
        return LLMResponse(
            content=answer.content,
            confidence=retrieval_confidence(query),
            latency_s=latency,
        )


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI via langchain-openai.

    Authentication: AZURE_OPENAI_API_KEY environment variable
      (or Azure AD token via DefaultAzureCredential)

    Required env vars / config:
      MTTR_AZURE_ENDPOINT    e.g. https://my-resource.openai.azure.com
      MTTR_AZURE_DEPLOYMENT  e.g. gpt-4o
      AZURE_OPENAI_API_KEY   set in environment

    Install: pip install langchain-openai
    """

    def __init__(self, config: ProviderConfig) -> None:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "Install langchain-openai: pip install langchain-openai"
            ) from exc

        self._llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            azure_deployment=config.azure_deployment,
            api_version=config.azure_api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def invoke(self, query: str, context: str = "") -> LLMResponse:
        from langchain_core.messages import HumanMessage
        t0 = time.perf_counter()
        answer = self._llm.invoke([HumanMessage(content=query)])
        latency = time.perf_counter() - t0
        return LLMResponse(
            content=answer.content,
            confidence=retrieval_confidence(query),
            latency_s=latency,
        )


# ── GCP Vertex AI ─────────────────────────────────────────────────────────────

class VertexAIProvider(BaseLLMProvider):
    """
    GCP Vertex AI (Gemini) via langchain-google-vertexai.

    Authentication: Application Default Credentials
      gcloud auth application-default login

    Required env vars / config:
      MTTR_GCP_PROJECT   GCP project ID
      MTTR_GCP_LOCATION  e.g. us-central1
      MTTR_MODEL_ID      e.g. gemini-1.5-pro-002

    Install: pip install langchain-google-vertexai
    """

    def __init__(self, config: ProviderConfig) -> None:
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise ImportError(
                "Install langchain-google-vertexai: pip install langchain-google-vertexai"
            ) from exc

        self._llm = ChatVertexAI(
            model_name=config.model_id,
            project=config.gcp_project,
            location=config.gcp_location,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    def invoke(self, query: str, context: str = "") -> LLMResponse:
        from langchain_core.messages import HumanMessage
        t0 = time.perf_counter()
        answer = self._llm.invoke([HumanMessage(content=query)])
        latency = time.perf_counter() - t0
        return LLMResponse(
            content=answer.content,
            confidence=retrieval_confidence(query),
            latency_s=latency,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_provider(config: ProviderConfig, seed: int = 42) -> BaseLLMProvider:
    """Instantiate the correct provider from config. Fails fast on bad config."""
    if config.kind == ProviderKind.MOCK:
        return MockProvider(config, seed=seed)
    if config.kind == ProviderKind.BEDROCK:
        return BedrockProvider(config)
    if config.kind == ProviderKind.AZURE_OPENAI:
        if not config.azure_endpoint or not config.azure_deployment:
            raise ValueError(
                "BedrockProvider requires azure_endpoint and azure_deployment in ProviderConfig"
            )
        return AzureOpenAIProvider(config)
    if config.kind == ProviderKind.VERTEX_AI:
        if not config.gcp_project:
            raise ValueError("VertexAIProvider requires gcp_project in ProviderConfig")
        return VertexAIProvider(config)
    raise ValueError(f"Unknown provider kind: {config.kind}")


