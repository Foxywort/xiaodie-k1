$ErrorActionPreference = "Stop"

. "E:\Duanwu\scripts\env.ps1"

python scripts\collect_ip_corpus.py --config configs\ip_rag_sources.yaml
python scripts\build_ip_rag_kb.py --config configs\ip_rag_sources.yaml
python scripts\build_ip_augmented_sft.py --rag-config configs\ip_rag_sources.yaml

Write-Host "IP RAG pipeline complete."
Write-Host "Cards: E:\Duanwu\data\ip_rag\ip_knowledge_cards.jsonl"
Write-Host "Index: E:\Duanwu\data\ip_rag\ip_rag_index.json"
