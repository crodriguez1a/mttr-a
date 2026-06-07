from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import MTTRSession


class CorpusTool:
    name = "search_corpus"
    description = (
        "Search the technical knowledge base for relevant documents. "
        "Returns the top 3 matching passages."
    )

    @property
    def claude_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents",
                    }
                },
                "required": ["query"],
            },
        }

    @property
    def gemini_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The search query to find relevant documents",
                    }
                },
                "required": ["query"],
            },
        }

    def execute(self, input_dict: dict, session: MTTRSession) -> str:
        import numpy as np

        from mttr_a.providers import _get_corpus_embs, _get_embed_model, _load_corpus

        query = input_dict.get("query", "")
        corpus_embs = _get_corpus_embs()
        docs = _load_corpus()
        model = _get_embed_model()

        q = model.encode([query], convert_to_numpy=True)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = (corpus_embs @ q.T).flatten()
        top_indices = scores.argsort()[::-1][:3]

        results = []
        top_doc_text = None
        for i, idx in enumerate(top_indices):
            doc = docs[idx]
            score = float(scores[idx])
            results.append(f"[{i+1}, relevance={score:.3f}]\n{doc}")
            if i == 0:
                top_doc_text = doc

        if top_doc_text:
            session.set_evidence(top_doc_text)

        return "\n\n".join(results)


class CalculatorTool:
    name = "calculate"
    description = "Safely evaluate a mathematical expression and return the numeric result."

    @property
    def claude_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate (e.g. '(43800 / 60) * 0.001')",  # noqa: E501
                    }
                },
                "required": ["expression"],
            },
        }

    @property
    def gemini_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "expression": {
                        "type": "STRING",
                        "description": "A mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        }

    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def execute(self, input_dict: dict, session: MTTRSession) -> str:
        expression = input_dict.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval(tree.body)
            return str(round(result, 6) if isinstance(result, float) else result)
        except Exception as exc:
            return f"Error: {exc}"

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._SAFE_OPS:
            return self._SAFE_OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._SAFE_OPS:
            return self._SAFE_OPS[type(node.op)](self._eval(node.operand))
        raise ValueError(f"Unsafe expression element: {ast.dump(node)}")


class ConcludeTool:
    name = "conclude"
    description = (
        "Submit your final answer and end the task. "
        "Call this when you have reached a complete, well-grounded conclusion."
    )

    @property
    def claude_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete final answer",
                    }
                },
                "required": ["answer"],
            },
        }

    @property
    def gemini_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "answer": {
                        "type": "STRING",
                        "description": "Your complete final answer",
                    }
                },
                "required": ["answer"],
            },
        }

    def execute(self, input_dict: dict, session: MTTRSession) -> str:
        return input_dict.get("answer", "")


DEMO_TOOLS: list = [CorpusTool(), CalculatorTool(), ConcludeTool()]
