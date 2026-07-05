# XiaoDie Project Plan

## Phase 1: Workstation Foundation

- Verify NVIDIA driver and CUDA PyTorch.
- Install Qwen-ASR runtime and data tooling.
- Prepare ASR dataset format and smoke tests.
- Keep model/cache files on E drive where possible.

## Phase 2: K1 Text-Input Prototype

- SSH into K1.
- Confirm audio output device with `aplay -l`.
- Pick a local TTS path that runs on K1.
- Build a command-line loop:
  `text input -> local response -> TTS wav -> headphone output`.

## Phase 3: ASR Adaptation

- Collect child-speech and classroom-noise samples with consent.
- Normalize audio to 16 kHz mono WAV.
- Start with LoRA/QLoRA or train on a larger GPU.
- Evaluate with WER/CER and child-specific intent accuracy.

## Phase 4: Edge Packaging

- Quantize ASR/LLM/TTS models according to K1 runtime support.
- Create a systemd service for the assistant.
- Add logging, watchdog restart, and offline fallback prompts.

## Phase 5: Story LLM Training

- Use a strict 3B text model as the first local training target.
- Train with 4bit QLoRA on RTX 4060 Ti 8GB.
- Keep every imported story row tagged with source and license.
- Merge LoRA, convert to GGUF, quantize, then benchmark on K1.
- Replace `generate_story()` in `board/xiaodie_story.py` after K1 inference is stable.

## Risk Notes

- 8GB VRAM is enough for environment validation and small LoRA experiments, not comfortable for full 1.7B ASR SFT.
- 8GB VRAM is also not realistic for full-parameter 3B LLM training; QLoRA is the practical local route.
- K1 has no current microphone input, so end-to-end voice input must wait for audio hardware.
- Children's voice data needs clear consent, minimization, and local storage discipline.
