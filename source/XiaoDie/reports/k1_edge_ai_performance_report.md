# K1 Edge AI Performance Report

Date: 2026-06-23

## Deployed Runtime

- Board: SpacemiT K1, Bianbu Linux, riscv64
- LLM: `xiaodie-story-1.5b:latest`
- GGUF: `/home/vicky/xiaodie/models/qwen2_5-1_5b-xiaodie-story-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Runtime: Ollama / llama.cpp on K1 CPU
- RAG: `/home/vicky/xiaodie/rag/ip_knowledge_cards.jsonl`
- TTS: Chaowen full, sherpa-onnx local offline inference
- TTS output: 3.5mm audio path via `aplay`

## Key Commands

Start competition runtime:

```bash
/home/vicky/xiaodie/llm/start_xiaodie_runtime.sh
```

Run RAG + LLM + streaming TTS:

```bash
cd /home/vicky/xiaodie
/home/vicky/xiaodie/llm/run_xiaodie_story_1_5b_tts.sh \
  --franchise peppa_pig \
  --query "佩奇和乔治整理玩具学会分享" \
  --age "4-6岁" \
  --style "睡前安抚" \
  --target-chars 100 \
  --max-new-tokens 96 \
  --ctx-size 1024 \
  --top-k 2 \
  --card-chars 80 \
  --output /home/vicky/xiaodie/reports/perf_e2e_after_runtime.md
```

## Measured Metrics

Pure local 1.5B LLM, warmed:

| Case | TTFT | Generation Speed | Prompt Eval |
| --- | ---: | ---: | ---: |
| short story | 2.98 s | 3.34 tokens/s | 87.69 tokens/s |
| medium story | 2.88 s | 3.38 tokens/s | 91.51 tokens/s |

RAG local LLM, cold prefix:

| Case | TTFT | Generation Speed | Prompt Eval |
| --- | ---: | ---: | ---: |
| Peppa RAG cold | 50.52 s | 3.56 tokens/s | 14.01 tokens/s |

RAG local LLM, cached prefix:

| Case | TTFT | Generation Speed | Prompt Eval |
| --- | ---: | ---: | ---: |
| Peppa RAG cached | 4.81 s | 3.20 tokens/s | 158.53 tokens/s |

End-to-end with Chaowen TTS running:

| Case | TTFT | Generation Speed | Prompt Eval |
| --- | ---: | ---: | ---: |
| RAG + LLM + streaming TTS | 5.34-5.74 s | 1.64-1.67 tokens/s | 131-141 tokens/s |

Chaowen full TTS after daemon optimization:

| Input | First Audio Chunk | Audio Duration | RTF |
| --- | ---: | ---: | ---: |
| `一天晚上，佩奇正在整理玩具。` | 4.70 s | 2.77 s | 1.84 |
| `乔治也来帮忙。` | 2.38 s | 1.35 s | 1.77 |

## Optimizations Applied

- Converted 1.5B LoRA model to GGUF and quantized to `Q4_K_M`.
- Registered the model as `xiaodie-story-1.5b:latest` in Ollama.
- Reordered RAG prompt so stable policy/cards come before dynamic user query, enabling prefix-cache reuse.
- Added `/home/vicky/xiaodie/llm/start_xiaodie_runtime.sh` to prewarm the LLM and Peppa RAG prefix before demo.
- Lowered default generation temperature for more stable story adherence.
- Reduced TTS service threads from 6 to 4 and started TTS with lower priority.
- Changed Chaowen daemon segmentation to dispatch the first complete sentence immediately.

## Current Bottlenecks

- First cold RAG prompt is still slow. The demo should run `start_xiaodie_runtime.sh` before judging.
- When Chaowen full synthesizes concurrently, LLM generation speed drops from about 3.2 tokens/s to about 1.6 tokens/s.
- Chaowen full quality is good, but first audio chunk still takes several seconds on K1.
- 1.5B story quality is usable for performance demo, but still weaker than the 4B model and can occasionally drift. Distillation remains the next quality step.
