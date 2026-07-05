import argparse

import torch
from qwen_asr import Qwen3ASRModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path or repo id.")
    parser.add_argument("--audio", required=True, help="Audio path, URL, or base64 data URL.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=1,
    )

    result = model.transcribe(
        args.audio,
        context=args.context,
        language=args.language,
        return_time_stamps=False,
    )[0]
    print(result.text)


if __name__ == "__main__":
    main()
