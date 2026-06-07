"""
Test suite for mttr_a.sdk — MTTRSession, tools, and adapter schemas.

Coverage
--------
  mttr_a/sdk/session.py       — MTTRSession: record_step, set_evidence, episode,
                                drift detection, pre-evidence confidence, step_confidences
  mttr_a/sdk/tools.py         — CalculatorTool: safe eval, unsafe rejection, edge cases
                                ConcludeTool: answer passthrough
                                CorpusTool: schemas, execute (mocked corpus)
  mttr_a/sdk/adapters/claude  — ClaudeAdapter.run(): full loop with mocked Anthropic client
  mttr_a/sdk/adapters/gemini  — GeminiAdapter.run(): full loop with mocked Gemini client
  mttr_a/sdk/__init__.py      — public re-exports
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mttr_a.sdk import ClaudeAdapter, GeminiAdapter, MTTRSession, StepRecord
from mttr_a.sdk.session import ToolRecord
from mttr_a.sdk.tools import DEMO_TOOLS, CalculatorTool, ConcludeTool, CorpusTool
from mttr_a_simulation import Episode

# ── Fixtures ───────────────────────────────────────────────────────────────────

FAKE_DOC = "distributed systems use consensus algorithms for fault tolerance"
FAKE_EMB = np.array([0.6, 0.8], dtype=np.float32)  # already normalized


def _make_session(task: str = "explain consensus algorithms") -> MTTRSession:
    return MTTRSession(task=task, run_id=0, drift_threshold=0.6)


def _seed_grounding(session: MTTRSession) -> None:
    """Directly set the evidence embedding without hitting the embed model."""
    session._evidence_emb = FAKE_EMB


def _patch_grounding(confidence: float = 0.75):
    """Seed grounding and patch step_groundedness — no disk/network access."""
    return patch("mttr_a.sdk.session.step_groundedness", return_value=confidence)


# ── MTTRSession ───────────────────────────────────────────────────────────────

class TestMTTRSessionInit:
    def test_initial_state(self):
        s = _make_session()
        assert s.drift_detected is False
        assert s.recovered is False
        assert s.latest_confidence is None
        assert s.step_confidences == []
        assert s.steps == []

    def test_run_id_stored(self):
        s = MTTRSession("task", run_id=7)
        assert s._run_id == 7


class TestMTTRSessionRecordStep:
    def test_no_confidence_before_grounding_set(self):
        s = _make_session()
        rec = s.record_step("some reasoning")
        assert rec.confidence is None
        assert rec.drift is False

    def test_confidence_scored_after_grounding_set(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.75):
            rec = s.record_step("some reasoning")
        assert rec.confidence == 0.75

    def test_record_returns_step_record(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.8):
            rec = s.record_step("reasoning text", latency_s=0.5, tool_called="search_corpus")
        assert isinstance(rec, StepRecord)
        assert rec.confidence == 0.8
        assert rec.latency_s == 0.5
        assert rec.tool_called == "search_corpus"
        assert rec.drift is False
        assert rec.turn == 0

    def test_step_index_increments(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.8):
            s.record_step("a")
            s.record_step("b")
            s.record_step("c")
        assert len(s.steps) == 3
        assert s.steps[2].turn == 2

    def test_reasoning_excerpt_truncated_to_200(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.75):
            rec = s.record_step("x" * 500)
        assert len(rec.reasoning_excerpt) == 200

    def test_step_confidences_property(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.8, 0.7, 0.55]):
            s.record_step("a")
            s.record_step("b")
            s.record_step("c")
        assert s.step_confidences == [0.8, 0.7, 0.55]

    def test_latest_confidence_reflects_last_step(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.9, 0.4]):
            s.record_step("high")
            s.record_step("low")
        assert s.latest_confidence == 0.4

    def test_latest_confidence_none_when_no_grounding(self):
        s = _make_session()
        s.record_step("no evidence yet")
        assert s.latest_confidence is None


class TestMTTRSessionDriftDetection:
    def test_no_drift_above_threshold(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.7):
            rec = s.record_step("grounded reasoning")
        assert rec.drift is False
        assert s.drift_detected is False

    def test_drift_below_threshold(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.3):
            rec = s.record_step("drifting reasoning")
        assert rec.drift is True
        assert s.drift_detected is True

    def test_no_drift_without_grounding(self):
        # Cannot detect drift if no evidence has been retrieved
        s = _make_session()
        rec = s.record_step("reasoning with no evidence")
        assert rec.drift is False
        assert s.drift_detected is False

    def test_drift_exactly_at_threshold_is_not_drift(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", return_value=0.6):
            rec = s.record_step("borderline")
        assert rec.drift is False

    def test_drift_flag_sticky_once_set(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.9]):
            s.record_step("drift")
            s.record_step("recovery")
        assert s.drift_detected is True

    def test_t_fault_set_on_first_drift(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.8, 0.3]):
            s.record_step("fine")
            t_before = time.perf_counter()
            s.record_step("drift")
            t_after = time.perf_counter()
        assert s._t_fault is not None
        assert t_before <= s._t_fault <= t_after

    def test_t_fault_not_overwritten_by_second_drift(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.2]):
            s.record_step("drift 1")
            first_t_fault = s._t_fault
            s.record_step("drift 2")
        assert s._t_fault == first_t_fault


class TestMTTRSessionSetEvidence:
    def test_sets_evidence_embedding(self):
        s = _make_session()
        new_emb = np.array([[0.5, 0.5, 0.707]], dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = new_emb

        with patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            s.set_evidence("machine learning uses gradient descent for optimization")

        assert s._evidence_emb is not None

    def test_l2_normalization_applied(self):
        s = _make_session()
        raw_emb = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5

        mock_model = MagicMock()
        mock_model.encode.return_value = raw_emb

        with patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            s.set_evidence("some doc")

        assert abs(np.linalg.norm(s._evidence_emb) - 1.0) < 1e-5


class TestMTTRSessionAutoRecovery:
    def test_recovery_detected_when_confidence_returns_above_tau(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.8]):
            s.record_step("drift step")
            rec = s.record_step("recovery step")
        assert s.recovered is True
        assert rec.recovered is True
        assert s._t_recovered is not None

    def test_recovery_not_detected_without_prior_drift(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.9):
            s.record_step("fine step")
        assert s.recovered is False

    def test_recovery_flag_not_set_twice(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.8, 0.2, 0.9]):
            s.record_step("drift")
            s.record_step("recovery")
            first_t = s._t_recovered
            s.record_step("drift again")
            s.record_step("above tau again")
        assert s._t_recovered == first_t


class TestMTTRSessionEpisode:
    def test_episode_returns_Episode_type(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.75):
            s.record_step("reasoning")
        assert isinstance(s.episode(), Episode)

    def test_episode_no_drift(self):
        s = _make_session("test query")
        _seed_grounding(s)
        with _patch_grounding(0.8):
            s.record_step("step A")
            s.record_step("step B")
        ep = s.episode()
        assert ep.drift_detected is False
        assert ep.delta_t == 0.0
        assert ep.run_id == 0
        assert ep.query == "test query"

    def test_episode_with_drift_has_nonnegative_delta_t(self):
        # t_execute is wall-clock time between real inference calls — always >= 0.
        # In tests the two record_step calls are microseconds apart so we only
        # assert the sign and that drift was recorded.
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.8]):
            s.record_step("drift step")
            s.record_step("recovery step")
        ep = s.episode()
        assert ep.drift_detected is True
        assert ep.delta_t >= 0.0

    def test_episode_step_confidences_tuple(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.7, 0.5, 0.9]):
            s.record_step("a")
            s.record_step("b")
            s.record_step("c")
        ep = s.episode()
        assert isinstance(ep.step_confidences, tuple)
        assert len(ep.step_confidences) == 3

    def test_episode_confidence_is_min_of_steps(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.9, 0.4, 0.8]):
            s.record_step("a")
            s.record_step("b")
            s.record_step("c")
        ep = s.episode()
        assert ep.confidence == round(0.4, 4)

    def test_episode_query_truncated_at_80_chars(self):
        s = MTTRSession("q" * 200)
        _seed_grounding(s)
        with _patch_grounding(0.75):
            s.record_step("step")
        assert len(s.episode().query) <= 80

    def test_episode_reflex_mode_none_when_drift_no_recovery(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.2):
            s.record_step("drift")
        assert s.episode().reflex_mode is None

    def test_episode_reflex_mode_none_when_no_drift(self):
        s = _make_session()
        _seed_grounding(s)
        with _patch_grounding(0.9):
            s.record_step("grounded")
        assert s.episode().reflex_mode is None

    def test_episode_reflex_mode_natural_when_recovered(self):
        s = _make_session()
        _seed_grounding(s)
        with patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.3, 0.8]):
            s.record_step("drift")
            s.record_step("naturally recovered")
        assert s.episode().reflex_mode == "natural"


class TestMTTRSessionToolCalls:
    def test_record_tool_call_stored(self):
        s = _make_session()
        s.record_tool_call(turn=0, tool_name="search_corpus", latency_s=0.42)
        assert len(s.tool_calls) == 1
        assert s.tool_calls[0].tool_name == "search_corpus"
        assert s.tool_calls[0].latency_s == 0.42
        assert s.tool_calls[0].turn == 0

    def test_record_tool_call_returns_tool_record(self):
        s = _make_session()
        rec = s.record_tool_call(turn=1, tool_name="calculate", latency_s=0.01)
        assert isinstance(rec, ToolRecord)

    def test_multiple_tool_calls_accumulated(self):
        s = _make_session()
        s.record_tool_call(0, "search_corpus", 0.3)
        s.record_tool_call(1, "calculate", 0.01)
        s.record_tool_call(2, "search_corpus", 0.25)
        assert len(s.tool_calls) == 3


# ── CalculatorTool ────────────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()
        self.session = MagicMock()

    def _calc(self, expression: str) -> str:
        return self.tool.execute({"expression": expression}, self.session)

    # basic operations
    def test_addition(self):
        assert self._calc("2 + 3") == "5"

    def test_subtraction(self):
        assert self._calc("10 - 4") == "6"

    def test_multiplication(self):
        assert self._calc("6 * 7") == "42"

    def test_division(self):
        assert self._calc("10 / 4") == "2.5"

    def test_power(self):
        assert self._calc("2 ** 10") == "1024"

    def test_unary_minus(self):
        assert self._calc("-5") == "-5"

    def test_unary_plus(self):
        assert self._calc("+3") == "3"

    def test_nested_expression(self):
        result = self._calc("(43800 / 60) * 0.001")
        assert abs(float(result) - 0.73) < 0.001

    def test_float_result_rounded_to_6(self):
        result = self._calc("1 / 3")
        assert result == "0.333333"

    def test_integer_result_not_rounded(self):
        assert self._calc("4 * 5") == "20"

    # unsafe inputs
    def test_rejects_function_call(self):
        result = self._calc("__import__('os').system('ls')")
        assert result.startswith("Error")

    def test_rejects_string_literal(self):
        result = self._calc("'hello'")
        assert result.startswith("Error")

    def test_rejects_name_access(self):
        result = self._calc("x + 1")
        assert result.startswith("Error")

    def test_empty_expression_returns_error(self):
        result = self._calc("")
        assert result.startswith("Error")

    def test_missing_key_uses_empty_string(self):
        result = self.tool.execute({}, self.session)
        assert result.startswith("Error")

    def test_schemas_have_required_fields(self):
        c = self.tool.claude_schema
        assert c["name"] == "calculate"
        assert "expression" in c["input_schema"]["properties"]
        g = self.tool.gemini_schema
        assert "expression" in g["parameters"]["properties"]


# ── ConcludeTool ──────────────────────────────────────────────────────────────

class TestConcludeTool:
    def setup_method(self):
        self.tool = ConcludeTool()
        self.session = MagicMock()

    def test_returns_answer(self):
        result = self.tool.execute({"answer": "42 is the answer"}, self.session)
        assert result == "42 is the answer"

    def test_missing_answer_returns_empty(self):
        result = self.tool.execute({}, self.session)
        assert result == ""

    def test_claude_schema_name(self):
        assert self.tool.claude_schema["name"] == "conclude"

    def test_gemini_schema_name(self):
        assert self.tool.gemini_schema["name"] == "conclude"

    def test_answer_required_in_schemas(self):
        assert "answer" in self.tool.claude_schema["input_schema"]["properties"]
        assert "answer" in self.tool.gemini_schema["parameters"]["properties"]


# ── CorpusTool ────────────────────────────────────────────────────────────────

class TestCorpusToolSchema:
    def setup_method(self):
        self.tool = CorpusTool()

    def test_name(self):
        assert self.tool.name == "search_corpus"

    def test_claude_schema_structure(self):
        s = self.tool.claude_schema
        assert s["name"] == "search_corpus"
        assert "query" in s["input_schema"]["properties"]
        assert "query" in s["input_schema"]["required"]

    def test_gemini_schema_structure(self):
        s = self.tool.gemini_schema
        assert s["name"] == "search_corpus"
        assert "query" in s["parameters"]["properties"]

    def test_gemini_schema_type_uppercase(self):
        s = self.tool.gemini_schema
        assert s["parameters"]["type"] == "OBJECT"
        assert s["parameters"]["properties"]["query"]["type"] == "STRING"


class TestCorpusToolExecute:
    def setup_method(self):
        self.tool = CorpusTool()

    def test_returns_top_docs_string(self):
        fake_docs = ["doc alpha", "doc beta", "doc gamma", "doc delta"]
        fake_embs = np.eye(4, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 0, 0, 0]], dtype=np.float32)

        session = MagicMock()

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=fake_docs), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            result = self.tool.execute({"query": "alpha"}, session)

        assert "doc alpha" in result
        assert "[1," in result

    def test_calls_set_evidence_with_top_doc(self):
        fake_docs = ["doc alpha", "doc beta", "doc gamma"]
        fake_embs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 0, 0]], dtype=np.float32)

        session = MagicMock()

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=fake_docs), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            self.tool.execute({"query": "alpha"}, session)

        session.set_evidence.assert_called_once_with("doc alpha")

    def test_returns_at_most_3_results(self):
        fake_docs = [f"doc {i}" for i in range(10)]
        fake_embs = np.eye(10, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1] + [0] * 9], dtype=np.float32)

        session = MagicMock()

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=fake_docs), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            result = self.tool.execute({"query": "anything"}, session)

        assert result.count("[1,") + result.count("[2,") + result.count("[3,") == 3


# ── DEMO_TOOLS ────────────────────────────────────────────────────────────────

class TestDemoTools:
    def test_all_three_tools_present(self):
        names = {t.name for t in DEMO_TOOLS}
        assert names == {"search_corpus", "calculate", "conclude"}

    def test_all_have_claude_schema(self):
        for t in DEMO_TOOLS:
            assert "name" in t.claude_schema
            assert "input_schema" in t.claude_schema

    def test_all_have_gemini_schema(self):
        for t in DEMO_TOOLS:
            assert "name" in t.gemini_schema
            assert "parameters" in t.gemini_schema


# ── Public SDK exports ────────────────────────────────────────────────────────

class TestSDKPublicExports:
    def test_mttr_session_importable(self):
        from mttr_a.sdk import MTTRSession as MS
        assert MS is MTTRSession

    def test_step_record_importable(self):
        from mttr_a.sdk import StepRecord as SR
        assert SR is StepRecord

    def test_adapters_importable(self):
        from mttr_a.sdk import ClaudeAdapter as CA
        from mttr_a.sdk import GeminiAdapter as GA
        assert CA is ClaudeAdapter
        assert GA is GeminiAdapter


# ── Adapter schema smoke tests (no API calls) ─────────────────────────────────

class TestClaudeAdapterSchema:
    def test_instantiable(self):
        assert ClaudeAdapter() is not None

    def test_run_raises_import_error_if_anthropic_missing(self):
        import sys
        real = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore
        try:
            with pytest.raises((ImportError, TypeError)):
                ClaudeAdapter().run("task", api_key="fake")
        finally:
            if real is None:
                del sys.modules["anthropic"]
            else:
                sys.modules["anthropic"] = real


class TestGeminiAdapterSchema:
    def test_instantiable(self):
        assert GeminiAdapter() is not None

    def test_run_raises_import_error_if_google_missing(self):
        import sys
        real = sys.modules.get("google.genai")
        sys.modules["google.genai"] = None  # type: ignore
        try:
            with pytest.raises((ImportError, TypeError, AttributeError)):
                GeminiAdapter().run("task", api_key="fake")
        finally:
            if real is None:
                del sys.modules["google.genai"]
            else:
                sys.modules["google.genai"] = real


# ── Helpers for adapter loop mocking ─────────────────────────────────────────

def _make_claude_text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b

def _make_claude_tool_block(name: str, input_dict: dict, block_id: str = "tu_1"):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = input_dict
    b.id = block_id
    return b

def _make_claude_response(stop_reason: str, *content_blocks):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = list(content_blocks)
    return r

def _make_gemini_candidate(text: str, fn_calls=None):
    candidate = MagicMock()
    parts = []
    if text:
        p = MagicMock()
        p.text = text
        p.function_call = None
        parts.append(p)
    for fc in (fn_calls or []):
        p = MagicMock()
        p.text = None
        p.function_call = fc
        parts.append(p)
    candidate.content.parts = parts
    return candidate

def _make_gemini_fn_call(name: str, args: dict):
    fc = MagicMock()
    fc.name = name
    fc.args = args
    return fc

def _make_gemini_response(candidate):
    r = MagicMock()
    r.candidates = [candidate]
    return r


# ── ClaudeAdapter full loop tests ────────────────────────────────────────────

class TestClaudeAdapterLoop:
    """Full adapter.run() loop with mocked Anthropic client — no API calls."""

    def _run(self, responses, task="test task", tools=None, queue=None):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            return ClaudeAdapter().run(
                task=task,
                tools=tools or DEMO_TOOLS,
                api_key="fake",
                event_queue=queue,
            )

    def test_single_turn_end_turn(self):
        resp = _make_claude_response("end_turn", _make_claude_text_block("Done."))
        session = self._run([resp])
        assert len(session.steps) == 1
        assert session.steps[0].tool_called is None

    def test_conclude_tool_breaks_loop(self):
        resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Final answer."),
            _make_claude_tool_block("conclude", {"answer": "42"}),
        )
        session = self._run([resp])
        assert len(session.steps) == 1

    def test_search_corpus_sets_evidence(self):
        # Turn 0: search_corpus → Turn 1: conclude
        search_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Searching..."),
            _make_claude_tool_block("search_corpus", {"query": "consensus"}, "tu_1"),
        )
        conclude_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Based on evidence."),
            _make_claude_tool_block("conclude", {"answer": "done"}, "tu_2"),
        )
        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc about consensus"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", return_value=0.75):
            session = self._run([search_resp, conclude_resp])

        assert session._evidence_emb is not None
        assert len(session.steps) == 2
        assert session.steps[0].confidence is None   # before evidence
        assert session.steps[1].confidence == 0.75   # after evidence

    def test_drift_detection_across_turns(self):
        search_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Searching."),
            _make_claude_tool_block("search_corpus", {"query": "q"}, "tu_1"),
        )
        drift_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Drifting reasoning."),
            _make_claude_tool_block("conclude", {"answer": "x"}, "tu_2"),
        )
        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", return_value=0.35):
            session = self._run([search_resp, drift_resp])

        assert session.drift_detected is True
        ep = session.episode()
        assert ep.drift_detected is True

    def test_natural_recovery_observed(self):
        search_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Searching."),
            _make_claude_tool_block("search_corpus", {"query": "q"}, "tu_1"),
        )
        drift_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Drifting."),
            _make_claude_tool_block("calculate", {"expression": "1+1"}, "tu_2"),
        )
        recover_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Re-anchored."),
            _make_claude_tool_block("conclude", {"answer": "done"}, "tu_3"),
        )
        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.35, 0.78]):
            session = self._run([search_resp, drift_resp, recover_resp])

        assert session.drift_detected is True
        assert session.recovered is True
        ep = session.episode()
        assert ep.reflex_mode == "natural"
        assert ep.delta_t >= 0.0

    def test_step_events_emitted(self):
        from queue import Queue
        q = Queue()
        resp = _make_claude_response("end_turn", _make_claude_text_block("Done."))
        self._run([resp], queue=q)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        step_events = [e for e in events if e["type"] == "step"]
        assert len(step_events) == 1
        assert "confidence" in step_events[0]
        assert "drift" in step_events[0]

    def test_tool_result_events_emitted(self):
        from queue import Queue
        q = Queue()
        search_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Searching."),
            _make_claude_tool_block("search_corpus", {"query": "q"}, "tu_1"),
        )
        conclude_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Done."),
            _make_claude_tool_block("conclude", {"answer": "x"}, "tu_2"),
        )
        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            self._run([search_resp, conclude_resp], queue=q)

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_result"]
        # conclude breaks the loop before its tool_result is emitted — only search_corpus fires
        assert len(tool_events) == 1
        assert tool_events[0]["tool_name"] == "search_corpus"
        assert "latency_s" in tool_events[0]

    def test_unknown_tool_handled_gracefully(self):
        resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Calling unknown."),
            _make_claude_tool_block("nonexistent_tool", {}, "tu_1"),
        )
        conclude_resp = _make_claude_response(
            "tool_use",
            _make_claude_text_block("Done."),
            _make_claude_tool_block("conclude", {"answer": "x"}, "tu_2"),
        )
        session = self._run([resp, conclude_resp])
        assert len(session.steps) == 2

    def test_max_turns_respected(self):
        # Response keeps returning tool_use without conclude
        def looping_resp():
            r = _make_claude_response(
                "tool_use",
                _make_claude_text_block("Thinking."),
                _make_claude_tool_block("calculate", {"expression": "1+1"}, "tu_x"),
            )
            return r
        session = self._run([looping_resp() for _ in range(20)], tools=DEMO_TOOLS)
        assert len(session.steps) <= 12

    def test_episode_produced_correctly(self):
        resp = _make_claude_response("end_turn", _make_claude_text_block("Done."))
        session = self._run([resp])
        ep = session.episode()
        from mttr_a_simulation import Episode
        assert isinstance(ep, Episode)
        assert ep.query == "test task"


# ── GeminiAdapter full loop tests ────────────────────────────────────────────

class TestGeminiAdapterLoop:
    """Full adapter.run() loop with mocked Gemini client — no API calls."""

    def _run(self, candidates, task="test task", tools=None, queue=None):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_gemini_response(c) for c in candidates
        ]
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google = MagicMock()
        mock_google.genai = mock_genai
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            return GeminiAdapter().run(
                task=task,
                tools=tools or DEMO_TOOLS,
                api_key="fake",
                event_queue=queue,
            )

    def test_single_turn_no_function_calls(self):
        cand = _make_gemini_candidate("Final answer.", fn_calls=[])
        session = self._run([cand])
        assert len(session.steps) == 1
        assert session.steps[0].tool_called is None

    def test_conclude_breaks_loop(self):
        fc = _make_gemini_fn_call("conclude", {"answer": "done"})
        cand = _make_gemini_candidate("Concluding.", fn_calls=[fc])
        session = self._run([cand])
        assert len(session.steps) == 1

    def test_search_corpus_sets_evidence(self):
        fc_search = _make_gemini_fn_call("search_corpus", {"query": "consensus"})
        cand_search = _make_gemini_candidate("Searching.", fn_calls=[fc_search])
        fc_conclude = _make_gemini_fn_call("conclude", {"answer": "done"})
        cand_conclude = _make_gemini_candidate("Done.", fn_calls=[fc_conclude])

        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", return_value=0.72):
            session = self._run([cand_search, cand_conclude])

        assert session._evidence_emb is not None
        assert session.steps[0].confidence is None
        assert session.steps[1].confidence == 0.72

    def test_drift_detected(self):
        fc_search = _make_gemini_fn_call("search_corpus", {"query": "q"})
        cand_search = _make_gemini_candidate("Searching.", fn_calls=[fc_search])
        fc_conclude = _make_gemini_fn_call("conclude", {"answer": "x"})
        cand_conclude = _make_gemini_candidate("Drifting.", fn_calls=[fc_conclude])

        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", return_value=0.3):
            session = self._run([cand_search, cand_conclude])

        assert session.drift_detected is True

    def test_natural_recovery_observed(self):
        fc_search = _make_gemini_fn_call("search_corpus", {"query": "q"})
        cand_search = _make_gemini_candidate("Searching.", fn_calls=[fc_search])
        fc_calc = _make_gemini_fn_call("calculate", {"expression": "2+2"})
        cand_drift = _make_gemini_candidate("Drifting.", fn_calls=[fc_calc])
        fc_conclude = _make_gemini_fn_call("conclude", {"answer": "done"})
        cand_recover = _make_gemini_candidate("Re-anchored.", fn_calls=[fc_conclude])

        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model), \
             patch("mttr_a.sdk.session.step_groundedness", side_effect=[0.32, 0.81]):
            session = self._run([cand_search, cand_drift, cand_recover])

        assert session.drift_detected is True
        assert session.recovered is True
        assert session.episode().reflex_mode == "natural"

    def test_step_events_emitted(self):
        from queue import Queue
        q = Queue()
        cand = _make_gemini_candidate("Done.", fn_calls=[])
        self._run([cand], queue=q)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        step_events = [e for e in events if e["type"] == "step"]
        assert len(step_events) == 1

    def test_tool_result_events_emitted(self):
        from queue import Queue
        q = Queue()
        fc_search = _make_gemini_fn_call("search_corpus", {"query": "q"})
        cand_search = _make_gemini_candidate("Searching.", fn_calls=[fc_search])
        fc_conclude = _make_gemini_fn_call("conclude", {"answer": "x"})
        cand_conclude = _make_gemini_candidate("Done.", fn_calls=[fc_conclude])

        fake_embs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embs

        with patch("mttr_a.providers._get_corpus_embs", return_value=fake_embs), \
             patch("mttr_a.providers._load_corpus", return_value=["doc"]), \
             patch("mttr_a.providers._get_embed_model", return_value=mock_model):
            self._run([cand_search, cand_conclude], queue=q)

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_events) >= 1
        assert tool_events[0]["tool_name"] == "search_corpus"
