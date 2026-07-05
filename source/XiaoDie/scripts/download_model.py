import argparse
from pathlib import Path


def download_from_modelscope(model: str, output: Path, max_workers: int | None = None) -> str:
    from modelscope import snapshot_download

    return snapshot_download(model, local_dir=str(output), max_workers=max_workers)


def download_from_huggingface(model: str, output: Path) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, local_dir=str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output", default=r"E:\xiaodie_models\Qwen3-ASR-1.7B")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.provider == "modelscope":
        path = download_from_modelscope(args.model, output, args.max_workers)
    else:
        path = download_from_huggingface(args.model, output)

    print(f"downloaded: {path}")


if __name__ == "__main__":
    main()
