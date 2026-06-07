from __future__ import annotations

import time
from queue import Queue

from mttr_a.sdk.session import MTTRSession
from mttr_a.sdk.tools import DEMO_TOOLS


def _emit(q: Queue | None, event: dict) -> None:
    if q is not None:
        q.put(event)


class ClaudeAdapter:
    def run(
        self,
        task: str,
        tools: list | None = None,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        max_turns: int = 12,
        event_queue: Queue | None = None,
        run_id: int = 0,
    ) -> MTTRSession:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        if tools is None:
            tools = DEMO_TOOLS

        client = anthropic.Anthropic(api_key=api_key)
        session = MTTRSession(task=task, run_id=run_id)
        tool_map = {t.name: t for t in tools}
        claude_tools = [t.claude_schema for t in tools]

        messages: list[dict] = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            t0 = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=claude_tools,
                messages=messages,
            )
            latency = time.perf_counter() - t0

            reasoning_text = " ".join(
                block.text
                for block in response.content
                if block.type == "text"
            ).strip()

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_called = tool_use_blocks[0].name if tool_use_blocks else None
            is_conclude = any(b.name == "conclude" for b in tool_use_blocks)

            step = session.record_step(
                reasoning_text=reasoning_text or task,
                latency_s=latency,
                tool_called=tool_called,
            )

            _emit(event_queue, {
                "type": "step",
                "turn": turn,
                "confidence": step.confidence,
                "reasoning_excerpt": step.reasoning_excerpt,
                "tool_called": tool_called,
                "drift": step.drift,
                "recovered": step.recovered,
            })

            if response.stop_reason == "end_turn" or is_conclude:
                break

            if response.stop_reason != "tool_use":
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                tool = tool_map.get(block.name)
                t_tool = time.perf_counter()
                result_text = (
                    tool.execute(block.input, session) if tool else f"Unknown tool: {block.name}"
                )
                tool_latency = time.perf_counter() - t_tool
                session.record_tool_call(turn=turn, tool_name=block.name, latency_s=tool_latency)
                _emit(event_queue, {
                    "type": "tool_result",
                    "turn": turn,
                    "tool_name": block.name,
                    "latency_s": round(tool_latency, 3),
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        return session
