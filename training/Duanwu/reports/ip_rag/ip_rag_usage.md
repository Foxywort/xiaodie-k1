# IP RAG Usage

## Build Knowledge Base

```powershell
cd E:\Duanwu
. .\scripts\run_ip_rag_pipeline.ps1
```

## Put Authorized Files Here

Place licensed scripts, episode summaries, character bibles, subtitles, or brand guidelines under:

```text
E:\Duanwu\data\ip_authorized\barbapapa
E:\Duanwu\data\ip_authorized\paw_patrol
E:\Duanwu\data\ip_authorized\octonauts
E:\Duanwu\data\ip_authorized\my_little_pony
E:\Duanwu\data\ip_authorized\peppa_pig
```

Supported formats: `.txt`, `.md`, `.html`, `.json`, `.jsonl`, `.srt`, `.vtt`.

After adding files, rerun `run_ip_rag_pipeline.ps1`.

## Generate With RAG

Use base Qwen3-4B for stricter instruction following:

```powershell
cd E:\Duanwu
. .\scripts\env.ps1
python .\scripts\rag_generate_story.py --adapter none --franchise peppa_pig --query "小猪佩奇和弟弟乔治一起分享玩具" --age "4-5岁" --style "睡前安抚" --output E:\Duanwu\reports\ip_rag\manual_test.md
```

Available `--franchise` values:

- `barbapapa`
- `paw_patrol`
- `octonauts`
- `my_little_pony`
- `peppa_pig`

Use the story LoRA only when you want softer prose and can tolerate weaker grounding:

```powershell
python .\scripts\rag_generate_story.py --adapter E:\Duanwu\outputs\best_story_adapter_round2 --franchise paw_patrol --query "汪汪队莱德带着阿奇和毛毛帮助小朋友整理玩具"
```

Current recommendation: use `--adapter none` for IP/RAG stories.
