# Confidence Evaluation Prompt

## What this is

The prompt sent to a real LLM as a second call immediately after it answers the primary query. The model's response is parsed as a decimal number in [0, 1] and used as the confidence score for drift detection.

## Why a second call

Different LLM providers expose model uncertainty in incompatible ways — some provide log-probabilities (a mathematical measure of certainty for each generated word), others don't expose them at all, and the format varies between providers and model versions. A self-evaluation prompt produces a plain decimal number that works identically across AWS Bedrock, Azure OpenAI, and Google Cloud Vertex AI without any provider-specific code.

The cost is one additional LLM call per episode. In a benchmark, that doubles the number of calls. In production monitoring (where you are sampling, not running every episode), the overhead is negligible.

## How it is used

All three real providers (`BedrockProvider`, `AzureOpenAIProvider`, `VertexAIProvider`) in `mttr_a/providers.py` load this file and send it as a follow-up message after answering the primary query. The response is parsed by `_parse_confidence()`, which extracts the first decimal-looking number from the reply and clamps it to [0, 1].

`MockProvider` does not use this file — it generates a confidence score directly from the mock distribution.

## How to modify

The prompt can be reworded as long as it instructs the model to reply with a single number in [0, 1]. Constraints:

- Keep it short — it is sent on every episode and contributes to token usage
- Keep the output format requirement explicit — the parser looks for a bare decimal number
- Do not add multi-turn structure — the prompt is appended as a single human message after the answer

Example alternative: `Rate your confidence in your last answer from 0 to 1. Output the number only.`

## Cross-model uniformity

For valid comparison across providers, use the same prompt file for all runs. A different phrasing may produce systematically higher or lower confidence scores on the same queries, making the drift rate incomparable.
