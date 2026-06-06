# Reference Corpus

## What this is and why it exists

This corpus is the source of the **confidence signal** used by real LLM providers in the MTTR-A benchmark.

The paper (*Barak Or, "MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems," arXiv:2511.20663v5*) defines confidence as:

```
confidence = cos(query_embedding, top_retrieved_document_embedding)
```

That is: embed the query, find the most similar document in a reference corpus, and return the cosine similarity between the query and that document as the confidence score. High similarity means the query maps to a well-represented concept in the corpus; lower similarity signals the query has drifted toward unfamiliar territory — the same semantic drift the benchmark is designed to detect.

## How confidence is computed

For each episode, real providers compute confidence with three steps:

1. **Embed the query** — encode the query string with `all-MiniLM-L6-v2` (sentence-transformers).
2. **Retrieve the top document** — compute the dot product of the query embedding against all pre-embedded, L2-normalised corpus embeddings and take the maximum score.
3. **Return the cosine score** — `cos(query_emb, top_doc_emb)`, clamped to [0, 1].

Corpus embeddings are pre-computed and cached in memory on first use, so retrieval adds negligible latency after the first episode.

This is implemented in `mttr_a/providers.py` as `retrieval_confidence(query)`.

## Mock provider

The `MockProvider` does **not** use this corpus. It simulates the retrieval-confidence distribution with a Gaussian:

```
confidence ~ N(base_mu=0.65, base_sigma=0.15) + N(noise_mu=0.0, noise_sigma=0.05)
```

This approximates the distribution of `cos(query, top_doc)` scores a real retrieval pipeline would produce — roughly 30–40% of episodes fall below the 0.6 drift threshold, generating enough drift events for meaningful metric computation. No corpus is needed for mock runs.

Gaussian parameters are tunable in `data/mock_config/config.json`.

## Replacing the corpus for production use

To measure MTTR-A against a domain-specific system, replace `documents.txt` with documents representative of your production query space:

1. Write one document per line — 1–3 sentences each, covering the concepts your system is expected to reason about.
2. Lines starting with `#` are treated as comments and ignored.
3. Blank lines are ignored.
4. Aim for broad semantic coverage so that on-topic queries score high (≥ 0.6) and off-topic or drifting queries score lower.

The corpus is loaded and embedded lazily on first use. Restart the process (or clear the module-level cache) after replacing the file.

## Format requirements

```
# Comment lines are ignored.
First document text here. One to three sentences.
Second document text here.
# Another comment.
Third document text here.
```

- One document per line
- No blank lines between documents (blank lines are silently skipped)
- Lines starting with `#` are comments
- UTF-8 encoding

## Bundled corpus

`documents.txt` ships with 97 documents covering:
distributed systems, rate limiting, caching, queues, databases, algorithms, machine learning and AI, software engineering, networking, security, reliability/SRE, agent systems and LLM orchestration, general reasoning, and programming concepts.

This breadth ensures any technical reasoning query finds a semantically relevant match, producing confidence scores that faithfully reflect whether the query maps to a known concept.
