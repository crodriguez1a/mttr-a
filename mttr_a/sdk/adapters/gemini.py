from __future__ import annotations

import time
from queue import Queue

from mttr_a.sdk.session import MTTRSession
from mttr_a.sdk.tools import DEMO_TOOLS


def _emit(q: Queue | None, event: dict) -> None:
    if q is not None:
        q.put(event)


class GeminiAdapter:
    def run(
        self,
        task: str,
        tools: list | None = None,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        max_turns: int = 12,
        event_queue: Queue | None = None,
        run_id: int = 0,
    ) -> MTTRSession:
        try:
            import google.genai as genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("pip install google-genai") from exc

        if tools is None:
            tools = DEMO_TOOLS

        client = genai.Client(api_key=api_key)
        session = MTTRSession(task=task, run_id=run_id)
        tool_map = {t.name: t for t in tools}

        function_declarations = [t.gemini_schema for t in tools]
        gemini_tool = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(
            tools=[gemini_tool],
            temperature=0.0,
            max_output_tokens=1024,
        )

        contents: list = [
            types.Content(role="user", parts=[types.Part(text=task)])
        ]

        for turn in range(max_turns):
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            latency = time.perf_counter() - t0

            candidate = response.candidates[0]
            text_parts = [
                p.text for p in candidate.content.parts if p.text is not None
            ]
            reasoning_text = " ".join(text_parts).strip()

            fn_calls = [
                p.function_call
                for p in candidate.content.parts
                if p.function_call is not None
            ]
            tool_called = fn_calls[0].name if fn_calls else None
            is_conclude = any(fc.name == "conclude" for fc in fn_calls)

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

            if is_conclude or not fn_calls:
                break

            contents.append(candidate.content)

            fn_responses = []
            for fc in fn_calls:
                tool = tool_map.get(fc.name)
                t_tool = time.perf_counter()
                result = (
                    tool.execute(dict(fc.args), session) if tool else f"Unknown function: {fc.name}"
                )
                tool_latency = time.perf_counter() - t_tool
                session.record_tool_call(turn=turn, tool_name=fc.name, latency_s=tool_latency)
                _emit(event_queue, {
                    "type": "tool_result",
                    "turn": turn,
                    "tool_name": fc.name,
                    "latency_s": round(tool_latency, 3),
                })
                fn_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=fn_responses))

        return session
