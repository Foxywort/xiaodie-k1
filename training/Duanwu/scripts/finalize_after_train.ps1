param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Continue"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "E:\Duanwu\logs\round2_finalize_$timestamp.log"

function Write-FinalizeLog {
    param([string]$Text)
    $Text | Tee-Object -FilePath $log -Append
}

Write-FinalizeLog "Round-2 finalizer started at $(Get-Date -Format s)"
Write-FinalizeLog "Waiting for PIDs: $($WaitPids -join ', ')"

foreach ($waitPid in $WaitPids) {
    try {
        if (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
            Wait-Process -Id $waitPid
            Write-FinalizeLog "PID $waitPid exited."
        } else {
            Write-FinalizeLog "PID $waitPid is not running."
        }
    } catch {
        Write-FinalizeLog "Wait failed for PID ${waitPid}: $($_.Exception.Message)"
    }
}

try {
    Set-Location "E:\Duanwu"
    . "E:\Duanwu\scripts\env.ps1" *>> $log

    Write-FinalizeLog "Selecting best checkpoint..."
    python scripts\select_best_checkpoint.py --config configs\train.yaml *>> $log

    Write-FinalizeLog "Generating test stories..."
    python scripts\generate_story.py --config configs\train.yaml --keywords "月亮、勇气、分享" --style "睡前安抚" --age "4-5岁" --output "E:\Duanwu\reports\story_test_moon_courage.md" *>> $log
    python scripts\generate_story.py --config configs\train.yaml --keywords "小兔子、诚实、朋友" --style "温柔有趣" --age "5-6岁" --output "E:\Duanwu\reports\story_test_honesty_friend.md" *>> $log
    python scripts\generate_story.py --config configs\train.yaml --keywords "幼儿园、积木、合作" --style "日常陪伴" --age "4-5岁" --output "E:\Duanwu\reports\story_test_kindergarten_blocks.md" *>> $log

    Write-FinalizeLog "Finalizer completed at $(Get-Date -Format s)"
} catch {
    Write-FinalizeLog "Finalizer failed: $($_.Exception.Message)"
    exit 1
}
