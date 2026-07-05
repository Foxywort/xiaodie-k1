# XiaoDie LLM Training

This directory is for the local LLM that will replace the current template story generator on K1.

## Current Strategy

- Product target: local Chinese story assistant for kindergarten children.
- Training workstation: RTX 4060 Ti 8GB.
- Practical method: 3B base model + 4bit QLoRA + LoRA adapter.
- Default trainable base today: `HuggingFaceTB/SmolLM3-3B` because it is Apache-2.0 and loads as a text-only CausalLM in the current Windows environment.
- Latest strict-3B target to track next: `mistralai/Ministral-3-3B-Instruct-2512-BF16`, Apache-2.0. It currently needs a newer/cleaner model stack than the existing ASR environment.

Full-parameter pretraining or full-parameter SFT of a 3B model is not realistic on an 8GB 4060 Ti. QLoRA is the aggressive local path that can actually run.

## Check Environment

```powershell
.\scripts\env.ps1
python llm\scripts\check_train_env.py --smoke-4bit
```

## Prepare Data

The default generator creates project-owned Chinese instruction data for the XiaoDie story domain. It also supports appending local JSONL files, but local files should include license/source metadata before release use.

```powershell
.\scripts\env.ps1
python llm\scripts\prepare_story_dataset.py --count 50000 --eval-count 512
```

Output:

- `llm/data/processed/story_sft_train.jsonl`
- `llm/data/processed/story_sft_eval.jsonl`

Accepted local JSONL formats:

```json
{"instruction":"请根据关键词：月亮、勇气，讲一个故事。","output":"《月亮和小小办法》...","source":"licensed_source_name","license":"cc-by-4.0"}
```

or:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"source":"...","license":"..."}}
```

## Train

```powershell
.\scripts\env.ps1
python llm\scripts\train_qlora.py --config llm\configs\story_sft_smollm3_3b_qlora.yaml
```

The first run downloads several GB of model weights into `E:/xiaodie_models/huggingface`.

## Deployment Path

1. Train LoRA adapter on the workstation.
2. Merge adapter into the base model on the workstation.
3. Convert merged model to GGUF with `llama.cpp`.
4. Quantize to `Q4_K_M` or smaller for K1.
5. Run local inference on K1 and replace `generate_story()` in `board/xiaodie_story.py`.
6. Measure TTFT, tokens/s, memory, and audio latency.

## Data Policy

Do not scrape copyrighted bedtime-story sites for release training. Use public-domain, permissively licensed, or directly authorized data, and keep source/license metadata with every imported row.
