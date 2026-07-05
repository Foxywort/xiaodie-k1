$ErrorActionPreference = "Stop"

. "E:\Duanwu\scripts\env.ps1"

python scripts\collect_open_data.py --config configs\datasets.yaml
python scripts\clean_filter_dedupe.py --config configs\datasets.yaml
python scripts\build_sft.py --config configs\train.yaml

Write-Host "Duanwu data pipeline complete."
Write-Host "Reports:"
Write-Host "  E:\Duanwu\reports\license_report.md"
Write-Host "  E:\Duanwu\reports\data_stats.md"
Write-Host "  E:\Duanwu\reports\sft_build_report.md"
