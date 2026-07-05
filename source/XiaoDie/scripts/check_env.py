import importlib
import importlib.metadata
import platform
import sys


def check_import(name: str, module_name: str | None = None) -> None:
    module_name = module_name or name
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}: MISSING")
        return
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", version)
        print(f"{name}: OK {version}".rstrip())
    except Exception as exc:
        print(f"{name}: INSTALLED {version}, IMPORT_FAILED {type(exc).__name__}: {exc}")


def main() -> None:
    import torch

    print(f"python: {sys.version}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print(f"cuda smoke mean: {float(y.mean().cpu()):.6f}")

    checks = [
        ("transformers", None),
        ("datasets", None),
        ("accelerate", None),
        ("peft", None),
        ("librosa", None),
        ("soundfile", None),
        ("jiwer", None),
        ("qwen_asr", None),
        ("bitsandbytes", None),
        ("trl", None),
        ("PyYAML", "yaml"),
        ("huggingface_hub", None),
    ]
    for package_name, module_name in checks:
        check_import(package_name, module_name)


if __name__ == "__main__":
    main()
