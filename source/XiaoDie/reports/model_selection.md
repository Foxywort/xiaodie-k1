# Model Selection

Selected model: `Qwen/Qwen3-4B`

| Model | Role | License | Score | Decision | Notes |
|---|---|---|---:|---|---|
| Qwen/Qwen3-4B | generative | apache-2.0 | 1.0175 | selected | Preferred Chinese text-generation 4B base. |
| HuggingFaceTB/SmolLM3-3B | generative | apache-2.0 | 0.67 | candidate | Stable fallback on current Windows stack. |
| Qwen/Qwen3.5-4B | image-text-to-text | apache-2.0 |  | role=image-text-to-text is not generative | Newer but multimodal architecture; train only if stack supports it. |
| Qwen/Qwen3-Embedding-4B | embedding | apache-2.0 |  | role=embedding is not generative | Not a generative model; can be used for retrieval/scoring, not story generation. |
