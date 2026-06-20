import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
import torch

from vlm_model.vlm import VLMForCausalLM
from vlm_model.utils import IMAGE_TOKEN
from data.image_processing import load_and_process_image
from training.checkpoint import load_connector_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_vlm(
    config_path: str, connector_checkpoint: str, device: str = "cuda"
) -> VLMForCausalLM:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = VLMForCausalLM(config)
    load_connector_checkpoint(model.connector, connector_checkpoint)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from {connector_checkpoint}")
    return model


def build_conversation(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{IMAGE_TOKEN}\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def decode_response(tokenizer, output_ids: torch.LongTensor) -> str:
    response = tokenizer.decode(output_ids, skip_special_tokens=False)

    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]

    return response.strip()


def build_generate_kwargs(tokenizer, max_new_tokens: int, temperature: float) -> Dict:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": 0.9,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "pad_token_id": tokenizer.pad_token_id,
    }


def run_inference(
    model: VLMForCausalLM,
    image_path: str,
    prompt: str = "Describe this image in detail.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    pixel_values = load_and_process_image(image_path, model.image_processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)

    conversation = build_conversation(prompt)

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"

    encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        images=pixel_values,
        attention_mask=attention_mask,
        **build_generate_kwargs(tokenizer, max_new_tokens, temperature),
    )

    return decode_response(tokenizer, output_ids[0])


def get_prompt(sample: Dict) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "").replace(IMAGE_TOKEN, "").strip()
    return "Describe this image."


def get_expected(sample: Dict) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def batched(iterable: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def run_batched_dataset_eval(
    model: VLMForCausalLM,
    data_path: str,
    image_dir: str,
    output_jsonl: str,
    batch_size: int,
    max_samples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if max_samples is not None:
        samples = samples[:max_samples]

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch in batched(samples, batch_size):
            prompts = [get_prompt(sample) for sample in batch]
            conversations = [build_conversation(prompt) for prompt in prompts]
            images = torch.stack(
                [
                    load_and_process_image(
                        os.path.join(image_dir, sample["image"]),
                        model.image_processor,
                    )
                    for sample in batch
                ]
            ).to(device)

            encoded = tokenizer(
                conversations,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                **build_generate_kwargs(tokenizer, max_new_tokens, temperature),
            )

            for sample, prompt, generated_ids in zip(batch, prompts, output_ids):
                image_path = os.path.join(image_dir, sample["image"])
                row = {
                    "id": sample.get("id", ""),
                    "image": sample["image"],
                    "image_path": image_path,
                    "prompt": prompt,
                    "expected": get_expected(sample),
                    "prediction": decode_response(tokenizer, generated_ids),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    logger.info(f"Wrote {written} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to connector checkpoint dir")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset JSON for batched evaluation")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory for batched evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataset evaluation")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional dataset evaluation sample limit")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Path to write batched evaluation JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_vlm(args.config, args.checkpoint, args.device)

    if args.data:
        if not args.image_dir or not args.output_jsonl:
            parser.error("--data requires --image_dir and --output_jsonl")
        run_batched_dataset_eval(
            model=model,
            data_path=args.data,
            image_dir=args.image_dir,
            output_jsonl=args.output_jsonl,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        return

    if not args.image:
        parser.error("--image is required unless --data is provided")

    response = run_inference(
        model=model,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
