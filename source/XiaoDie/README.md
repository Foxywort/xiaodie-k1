# XiaoDie

XiaoDie is an edge AI storytelling assistant for kindergarten children.

Current hardware split:

- Training workstation: Windows 10, RTX 4060 Ti 8GB, CUDA driver 12.6.
- Target board: Spacemit K1 Linux board, about 6 TOPS.
- Current K1 I/O: text input first, headphone audio output via TTS.
- Later K1 I/O: microphone input -> ASR -> local LLM -> TTS.

## Current Status

Done on this workstation:

- Created Python venv: `E:\xiaodie_envs\xiaodie-qwenasr`
- Installed CUDA PyTorch: `torch 2.12.0+cu126`
- Verified GPU compute on `NVIDIA GeForce RTX 4060 Ti`
- Installed `qwen-asr==0.0.6`, `transformers`, `datasets`, `accelerate`, `peft`, `librosa`, `soundfile`, `jiwer`, `modelscope`

K1 SSH alias is configured and verified:

```powershell
ssh k1 "whoami && hostname && uname -a"
```

Known K1 board info:

- User: `vicky`
- SSH alias: `k1`
- OS: `Bianbu 2.3.3`
- Kernel: `Linux 6.6.63`
- Architecture: `riscv64`
- Audio output: PipeWire default output works; direct ALSA hardware can be busy because PipeWire owns the device.
- Current TTS: Sherpa ONNX `v1.13.2` with `vits-piper-zh_CN-huayan-medium`, fallback to `espeak-ng`.
- Current LLM training stack: `bitsandbytes`, `peft`, `accelerate`, and `trl` are installed; 4bit QLoRA smoke training has passed on the RTX 4060 Ti.

## Activate Environment

```powershell
.\scripts\env.ps1
python scripts\check_env.py
```

## Model Download

Download Qwen3-ASR-1.7B only when ready; it will consume several GB.

```powershell
.\scripts\env.ps1
python scripts\download_model.py --provider modelscope --model Qwen/Qwen3-ASR-1.7B --output E:\xiaodie_models\Qwen3-ASR-1.7B
```

## ASR Smoke Test

After downloading the model and preparing a WAV file:

```powershell
.\scripts\env.ps1
python scripts\qwen3_asr_infer.py --model E:\xiaodie_models\Qwen3-ASR-1.7B --audio data\asr\sample.wav --language Chinese
```

## Training Direction

The official Qwen3-ASR fine-tuning script is full-parameter SFT. On this RTX 4060 Ti 8GB machine, full fine-tuning of a 1.7B ASR model is expected to OOM. Use one of these paths:

1. Practical local path: LoRA/QLoRA with tiny batch size, gradient checkpointing, short audio clips, and frequent eval.
2. Stable path: train or adapt ASR on a larger GPU/cloud machine, then quantize/export for K1.
3. Product-first path: keep ASR off-board during early tests, finish K1 text-input -> local LLM -> TTS flow first.

Data format for ASR fine-tuning:

```jsonl
{"audio":"D:/datasets/xiaodie/audio/001.wav","text":"小朋友们，我们开始讲故事吧。","prompt":"Transcribe the audio in Chinese."}
```

## K1 First Milestone

Because K1 currently has no audio input device, first ship:

```text
keyboard text input -> local LLM/story policy -> TTS -> headphone output
```

Current story-machine command on K1:

```bash
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气
```

The current generator is an offline template engine for MVP validation. Its `generate_story()` function is the planned replacement point for a local LLM.

Then replace keyboard input with:

```text
microphone -> ASR -> local LLM/story policy -> TTS -> headphone output
```

See `board/README.md`.

## LLM Story Model

The next milestone is replacing the template story generator with a local LLM:

```powershell
.\scripts\env.ps1
python llm\scripts\check_train_env.py --smoke-4bit
python llm\scripts\prepare_story_dataset.py --count 50000 --eval-count 512
python llm\scripts\train_qlora.py --config llm\configs\story_sft_smollm3_3b_qlora.yaml
```

The current practical training path is 3B + 4bit QLoRA, not full-parameter 3B training. See `llm/README.md`.

## Chinese Story LLM Pipeline

This repository now includes a reproducible SFT + LoRA/QLoRA project for a Chinese children story model.

Important model note: `Qwen/Qwen3-Embedding-4B` is an embedding model, so it is not suitable as the story generator. The training pipeline records it as unsuitable and prefers a generative Qwen 4B model, currently `Qwen/Qwen3-4B`, with `HuggingFaceTB/SmolLM3-3B` kept as a lower-cost fallback.

Install or refresh the training environment:

```powershell
cd D:\Spacemit\XiaoDie
.\scripts\env.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
python scripts\check_env.py
```

Build the legal starter dataset and reports:

```powershell
python scripts\download_datasets.py --config configs\datasets.yaml
python scripts\clean_stories.py --config configs\datasets.yaml
python scripts\deduplicate.py
python scripts\build_jsonl.py --config configs\train.yaml
```

Outputs:

- `data/processed/chinese_children_stories.jsonl`
- `data/processed/sft_train.jsonl`
- `data/processed/sft_eval.jsonl`
- `reports/license_report.md`
- `reports/data_stats.md`

The default dataset config is conservative: it uses project-generated, license-safe seed stories and records open candidates. Large sources such as `adam89/TinyStoriesChinese` and `opencsg/chinese-cosmopedia` are disabled by default. To use them, edit `configs/datasets.yaml`, set the source to `enabled: true`, then run download with:

```powershell
python scripts\download_datasets.py --config configs\datasets.yaml --download-large
```

Always run smoke training before any long training:

```powershell
python scripts\train_sft_lora.py --config configs\train.yaml --smoke
python scripts\select_best_checkpoint.py --output-dir outputs\smoke-story-lora --best-dir outputs\best_story_adapter_smoke
```

Run the formal Qwen 4B QLoRA training after network and disk are ready:

```powershell
python scripts\train_sft_lora.py --config configs\train.yaml
```

Main formal outputs:

- `outputs/story-qwen-lora`
- `outputs/story-qwen-lora/checkpoint-*`
- `outputs/story-qwen-lora/training_config.json`
- `outputs/story-qwen-lora/loss_curve.png`

Evaluate a trained adapter:

```powershell
python scripts\evaluate_model.py --config configs\train.yaml --base-model Qwen/Qwen3-4B --adapter outputs\story-qwen-lora --limit 30 --max-new-tokens 512
```

Evaluate a merged local model:

```powershell
python scripts\evaluate_model.py --base-model llm\outputs\smollm3-3b-story-merged-fp16 --merged --limit 30 --max-new-tokens 320
```

Evaluation outputs:

- `reports/generated_samples.md`
- `reports/evaluation_report.md`

Merge LoRA weights into a standalone Hugging Face model:

```powershell
python scripts\merge_lora.py --base-model Qwen/Qwen3-4B --adapter outputs\story-qwen-lora --output-dir outputs\story-qwen-merged-fp16 --dtype float16 --device-map cpu
```

Export the merged model to GGUF and quantize for edge experiments:

```powershell
python tools\llama.cpp\convert_hf_to_gguf.py outputs\story-qwen-merged-fp16 --outfile outputs\story-qwen-f16.gguf --outtype f16
tools\llama.cpp-bin-b9667-win-cpu-x64\llama-quantize.exe outputs\story-qwen-f16.gguf outputs\story-qwen-Q4_K_M.gguf Q4_K_M
```

Current verified results on this machine:

- CUDA training environment passed on `NVIDIA GeForce RTX 4060 Ti`.
- Data pipeline produced 199 deduplicated legal starter stories.
- Smoke LoRA training passed and saved `outputs/smoke-story-lora`.
- Checkpoint selection passed and saved `outputs/best_story_adapter_smoke`.
- Baseline evaluation on the previous local `SmolLM3-3B` merged model generated 30 samples with heuristic average score `0.945`.

## References

- Qwen3-ASR official repository: https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR package: https://pypi.org/project/qwen-asr/
- PyTorch local install selector: https://pytorch.org/get-started/locally/
