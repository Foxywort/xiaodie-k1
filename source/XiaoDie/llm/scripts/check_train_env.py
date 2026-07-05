#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata
import platform
import sys


def gib(value: int) -> float:
    return value / 1024**3


def import_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
    except Exception:
        pass

    try:
        module = importlib.import_module(package)
    except Exception as exc:
        return f"missing ({type(exc).__name__}: {exc})"
    return getattr(module, "__version__", "installed")


def check_cuda() -> bool:
    import torch

    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    print(f"gpu: {props.name}")
    print(f"vram: {gib(props.total_memory):.2f} GiB")
    print(f"compute capability: {props.major}.{props.minor}")
    return True


def smoke_4bit() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
    )
    inputs = tokenizer("小蝶讲一个月亮故事。", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4)
    print("4bit smoke: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check XiaoDie LLM training environment.")
    parser.add_argument("--smoke-4bit", action="store_true", help="Download a tiny model and test 4bit GPU loading.")
    args = parser.parse_args()

    cuda_ok = check_cuda()
    for package in ["transformers", "datasets", "accelerate", "peft", "bitsandbytes", "trl"]:
        print(f"{package}: {import_version(package)}")

    if args.smoke_4bit:
        if not cuda_ok:
            raise SystemExit("CUDA is not available; 4bit smoke cannot run.")
        smoke_4bit()


if __name__ == "__main__":
    main()
