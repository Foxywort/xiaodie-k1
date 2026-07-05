# Qwen2.5 1.5B XiaoDie Training Summary

## Result

- Base model: `E:/Duanwu/models/Qwen2.5-1.5B-Instruct`
- Train data: `E:/Duanwu/data/processed/sft_train.jsonl`
- Eval data: `E:/Duanwu/data/processed/sft_eval.jsonl`
- RAG KB retained: `E:/Duanwu/data/ip_rag/`
- Config: `D:/Spacemit/XiaoDie/configs/train_qwen2_5_1_5b.yaml`
- Full adapter output: `E:/Duanwu/outputs/qwen2_5-1_5b-story-round3`
- Best adapter: `E:/Duanwu/outputs/best_story_adapter_qwen2_5_1_5b`
- Best checkpoint: `checkpoint-2000`
- Best eval loss: `1.882116436958313`
- Runtime: `571.2752s`

## Training Command

```powershell
cd D:\Spacemit\XiaoDie
. E:\Duanwu\scripts\env.ps1
python -u E:\Duanwu\scripts\train_round2_lora.py --config D:\Spacemit\XiaoDie\configs\train_qwen2_5_1_5b.yaml
python E:\Duanwu\scripts\select_best_checkpoint.py --config D:\Spacemit\XiaoDie\configs\train_qwen2_5_1_5b.yaml
```

## Quality Notes

- Plain story generation loads and runs.
- RAG path loads and retrieves the existing IP knowledge cards.
- Current 1.5B output quality is not production-ready for IP stories: it may ignore RAG facts and hallucinate unsafe or off-setting details.
- Recommended next step: keep this model as a low-latency candidate, but add stricter K1-side RAG prompting, entity guards, and output validation before deployment.
