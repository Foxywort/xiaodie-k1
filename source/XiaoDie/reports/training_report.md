# Training Run Report

Date: 2026-06-18

## Completed In This Run

- Environment check passed with CUDA on NVIDIA GeForce RTX 4060 Ti.
- Legal starter data was generated from `xiaodie_project_seed`.
- Metadata was downloaded for `hughxusu/chinese-kid-story`; the current snapshot is metadata-only and was not used as story text.
- Large external datasets remain disabled by default for disk/time safety.
- Cleaning, filtering, near-duplicate removal, and SFT formatting completed.
- Smoke LoRA training completed on 50 samples and 3 optimization steps.
- Adapter saving completed.
- Checkpoint selection completed.
- Baseline evaluation completed with the previous local merged `SmolLM3-3B` story model.

## Commands Run

```powershell
python scripts\check_env.py
python scripts\download_datasets.py --config configs\datasets.yaml
python scripts\clean_stories.py --config configs\datasets.yaml
python scripts\deduplicate.py
python scripts\build_jsonl.py --config configs\train.yaml
python scripts\train_sft_lora.py --config configs\train.yaml --smoke
python scripts\select_best_checkpoint.py --output-dir outputs\smoke-story-lora --best-dir outputs\best_story_adapter_smoke
python scripts\evaluate_model.py --base-model llm\outputs\smollm3-3b-story-merged-fp16 --merged --limit 30 --max-new-tokens 320
```

## Data Outputs

- `data/processed/chinese_children_stories.jsonl`
- `data/processed/sft_train.jsonl`
- `data/processed/sft_eval.jsonl`

## Model Outputs

- `outputs/smoke-story-lora`
- `outputs/best_story_adapter_smoke`

## Formal Training Status

Formal Qwen 4B QLoRA training is ready but was not launched in this run. It requires downloading the full `Qwen/Qwen3-4B` base model and should be started when the internet connection and disk budget are stable.

Command:

```powershell
python scripts\train_sft_lora.py --config configs\train.yaml
```

The expected formal adapter output is `outputs/story-qwen-lora`.

## Current Qwen 4B Long Run

Started on 2026-06-18 with the local ModelScope model directory:

- base_model: `E:\xiaodie_models\Qwen3-4B`
- adapter_output: `outputs/story-qwen-lora`
- max_seq_length: 256
- max_steps: 2000
- log_stdout: `outputs/logs/qwen_local_train_256_20260618_185525.out.log`
- log_stderr: `outputs/logs/qwen_local_train_256_20260618_185525.err.log`

Observed early training status:

- step: 25 / 2000
- loss trend: 4.6332 -> 3.0143
- GPU: 100% utilization, about 7.8GB VRAM used

The run is expected to take many hours on RTX 4060 Ti because Qwen3-4B QLoRA is near the 8GB VRAM limit on Windows.

## Completed Qwen 4B Run

The Qwen3-4B QLoRA long run completed successfully on 2026-06-18.

- final_step: 2000 / 2000
- final_epoch: 10.87
- train_runtime: 9875.87 seconds
- final_train_loss: 0.1001
- checkpoint_1800_eval_loss: 0.0259069223
- checkpoint_1900_eval_loss: 0.0258168783
- checkpoint_2000_eval_loss: 0.0254110787
- selected_best_checkpoint: `outputs/story-qwen-lora/checkpoint-2000`
- copied_best_adapter: `outputs/best_story_adapter`
- final_adapter: `outputs/story-qwen-lora`

## Post-Training Evaluation

Automatic heuristic evaluation was run on 30 story prompts.

- samples: 30
- average_score: 0.935
- unsafe_count: 0
- too_short_count: 0
- complete_structure_ratio: 0.433
- avg_chinese_ratio: 0.846

Reports:

- `reports/evaluation_report.md`
- `reports/generated_samples.md`
- `reports/manual_story_test_cleaned.md`

Quality notes:

- The adapter can generate safe Chinese children stories.
- It is heavily adapted to the small starter dataset and shows template repetition.
- Some raw generations include English words or preamble text; `scripts/generate_qwen_story.py` applies a stricter prompt and post-processing for the story-machine path.
- The next quality jump should come from more diverse legal data, not more epochs on the current small dataset.
