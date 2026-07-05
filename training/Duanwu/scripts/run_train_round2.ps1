$ErrorActionPreference = "Stop"

. "E:\Duanwu\scripts\env.ps1"
. "E:\Duanwu\scripts\prepare_base_model.ps1"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = "E:\Duanwu\logs\round2_train_$timestamp.out.log"
$stderr = "E:\Duanwu\logs\round2_train_$timestamp.err.log"

$p = Start-Process -FilePath powershell.exe `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        ". E:\Duanwu\scripts\env.ps1; python -u scripts\train_round2_lora.py --config configs\train.yaml"
    ) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

Write-Host "Round-2 training started."
Write-Host "PID: $($p.Id)"
Write-Host "STDOUT: $stdout"
Write-Host "STDERR: $stderr"
