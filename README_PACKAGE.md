# 小蝶 K1 端侧语音故事机

## 目录说明

- source/XiaoDie/：主工程源码。包含板端应用、Qt 原生界面、ASR/TTS/LLM 调度脚本、RAG 生成脚本、部署脚本、比赛文档和技术说明。
- 	raining/Duanwu/：第二轮训练工程。包含合法数据收集与清洗脚本、训练配置、清洗后的儿童故事数据、IP RAG 知识库、1.5B/4B LoRA 输出、训练日志和评估报告。
-  artifacts/deploy_artifacts/：最终部署资产。包含合并后的 Hugging Face fp16 模型、GGUF/f16/Q4/Q5 量化模型、TTS 下载包。
- board_runtime/k1_xiaodie/：从 K1 板卡 /home/vicky/xiaodie 拉取的实际运行目录。包含 ASR 模型、Chaowen/TTS 解压运行环境、板端 app、llm/rag/reports 等。
- demos/k1_story_audio_20260630/：今天从板卡导出的艾莎公主、小猪佩奇故事音频和对应文本。

## 最关键的部署模型

- 端侧首选 LLM：artifacts/deploy_artifacts/gguf/qwen2_5-1_5b-xiaodie-story-Q4_K_M.gguf
- 1.5B 合并模型：artifacts/deploy_artifacts/models/qwen2_5-1_5b-xiaodie-story-merged-fp16/
- 4B 备选模型：artifacts/deploy_artifacts/gguf/qwen3-4b-xiaodie-story-Q4_K_M.gguf、qwen3-4b-xiaodie-story-Q5_K_M.gguf
- RAG 知识库：	training/Duanwu/data/ip_rag/，板端副本在 board_runtime/k1_xiaodie/rag/
- ASR 运行模型：board_runtime/k1_xiaodie/asr/
- TTS 运行模型：board_runtime/k1_xiaodie/tts/vits-piper-zh_CN-chaowen-medium/
- Qt 界面源码：source/XiaoDie/deploy/k1_app/qt/main.cpp
- 板端主程序：source/XiaoDie/deploy/k1_app/xiaodie_button_story.py
- DeepSeek 流式故事 + TTS chunk 调度：source/XiaoDie/deploy/k1_llm/xiaodie_deepseek_tts_stream.py

## 复现入口

训练项目入口：

`powershell
cd training\Duanwu
. .\scripts\env.ps1
python .\scripts\train_round2_lora.py --config .\configs\train.yaml
`

本地 RAG 文本测试入口：

`powershell
cd training\Duanwu
. .\scripts\env.ps1
python .\scripts\rag_generate_story.py --adapter outputs\best_story_adapter_qwen2_5_1_5b --franchise peppa_pig --query "小猪佩奇和乔治一起分享玩具" --age "4-6岁" --style "睡前安抚" --target-chars 700
`

K1 板端运行入口：

`bash
cd /home/vicky/xiaodie
sudo env XIAODIE_BUTTON_GPIO=35 XIAODIE_BUTTON_ACTIVE=low /home/vicky/xiaodie/app/start_xiaodie_button_story.sh
`


## 校验文件

- PACKAGE_FILE_LIST.csv：包内完整文件清单。
- PACKAGE_TOP_LEVEL_SIZES.csv：顶层目录大小。
- IMPORTANT_HASHES.sha256：GGUF、LoRA adapter、ASR/TTS 关键文件哈希。
