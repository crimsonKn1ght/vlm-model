# Vision-Language Model (VLM) - Stage 1 Feature Alignment

A minimal, efficient implementation of a Vision-Language Model following the LLaVA architecture. This project trains only a lightweight MLP connector to align frozen CLIP vision features with a frozen LLM, achieving effective multimodal understanding with minimal computational cost.

## Overview

This implementation bridges a frozen CLIP vision encoder (`openai/clip-vit-large-patch14`) and a frozen instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`) using a simple 2-layer MLP connector. Only the connector (~3.9M parameters) is trained on image-caption pairs; both the vision encoder and LLM remain frozen throughout training.

**Key Design Principles:**
- **Simplicity**: No Q-Former, no cross-attention — just a linear projection with GELU
- **Efficiency**: Frozen models save memory; only train the connector
- **Proven approach**: Follows LLaVA Stage 1 alignment, a well-validated recipe

## Architecture

```
Image (3, 224, 224)
    ↓
[CLIP ViT-L/14 — Frozen] → 256 patch tokens (B, 256, 1024)
    ↓
[MLP Connector — Trainable] → (B, 256, 1536)
    ↓ (concatenated with)
Text Embeddings (B, T, 1536) from LLM embedding table
    ↓
[Qwen2.5-1.5B-Instruct — Frozen] → Loss
```

**Training objective**: Next-token prediction on image-text pairs. Visual tokens are masked out of the loss; only the caption tokens contribute to gradient updates on the connector.

## Installation

```bash
# Clone or navigate to the project directory
cd vlm

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.1.0
- CUDA 11.8+ (for GPU training)
- 12GB+ GPU memory (tested on RTX 3060, A100)

## Quick Start

### 1. Prepare Data

Download the LLaVA-Pretrain dataset or use your own image-text pairs in LLaVA format:

```json
[
  {
    "id": "...",
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": "A detailed caption here."}
    ]
  },
  ...
]
```

### 2. Configure Training

Edit `configs/pretrain_stage1.yaml`:

```yaml
data:
  train_data_path: "/path/to/blip_laion_cc_sbu_558k.json"  # or your dataset
  image_dir: "/path/to/images"
  max_length: 2048

training:
  output_dir: "./checkpoints/pretrain-stage1"
  per_device_batch_size: 8
  gradient_accumulation_steps: 32  # effective batch = 256
  num_epochs: 1
```

### 3. Train

```bash
# Single GPU
python train.py --config configs/pretrain_stage1.yaml

# Multi-GPU with accelerate
accelerate launch train.py --config configs/pretrain_stage1.yaml
```

Training logs appear in stdout. Checkpoints are saved every 500 steps to `./checkpoints/pretrain-stage1/`.

### RTX 6000 Ada Pod Workflow

For a CUDA pod, install dependencies and verify GPU visibility:

```bash
cd /workspace/vlm

python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

For basic connector training without the 558K image export, download a compact subset and train the lighter config:

```bash
python scripts/download_llava_recap.py \
  --max-samples 2048 \
  --output-dir datasets/llava_recap_small \
  --jpeg-quality 85 \
  --overwrite

python train.py --config configs/basic_connector_small.yaml

python inference.py \
  --config configs/basic_connector_small.yaml \
  --checkpoint checkpoints/basic-connector-small \
  --data datasets/llava_recap_small/train.json \
  --image_dir datasets/llava_recap_small/images \
  --batch_size 16 \
  --max_samples 32 \
  --max_new_tokens 80 \
  --temperature 0 \
  --output_jsonl outputs/basic-connector-small-eval.jsonl
```

`configs/basic_connector_small.yaml` uses `openai/clip-vit-base-patch32` and `Qwen/Qwen2.5-0.5B-Instruct`, so it is much lighter than the default CLIP-L/Qwen-1.5B setup. Increase or decrease `--max-samples` for the disk/runtime budget you actually want.

Run a fast batched smoke train and write batched qualitative generations:

```bash
python train.py --config configs/pod_smoke.yaml

python inference.py \
  --config configs/pod_smoke.yaml \
  --checkpoint checkpoints/pod-smoke \
  --data datasets/llava_recap_558k/train.json \
  --image_dir datasets/llava_recap_558k/images \
  --batch_size 16 \
  --max_samples 16 \
  --max_new_tokens 80 \
  --temperature 0 \
  --output_jsonl outputs/pod-smoke-eval.jsonl
```

Run the full one-epoch connector alignment pass and sample batched evaluation:

```bash
python train.py --config configs/pod_rtx6000ada.yaml

python inference.py \
  --config configs/pod_rtx6000ada.yaml \
  --checkpoint checkpoints/stage1-rtx6000ada \
  --data datasets/llava_recap_558k/train.json \
  --image_dir datasets/llava_recap_558k/images \
  --batch_size 16 \
  --max_samples 32 \
  --max_new_tokens 80 \
  --temperature 0 \
  --output_jsonl outputs/stage1-sample-eval.jsonl
```

`--checkpoint` can point either to a specific `checkpoint-N` directory or to its parent output directory; the latest checkpoint is resolved automatically.

### 4. Inference

```bash
python inference.py \
  --config configs/pretrain_stage1.yaml \
  --checkpoint ./checkpoints/pretrain-stage1/checkpoint-500 \
  --image /path/to/image.jpg \
  --prompt "What is in this image?"
```

## Project Structure

```
vlm/
├── train.py                        # Training entry point
├── inference.py                    # Inference script
├── requirements.txt                # Dependencies
├── configs/
│   └── pretrain_stage1.yaml        # Hyperparameter config
├── vlm_model/
│   ├── utils.py                    # Constants, helper functions
│   ├── connector.py                # MLP projection layer (trainable)
│   ├── vision_encoder.py           # CLIP ViT-L/14 wrapper
│   ├── language_model.py           # Qwen2.5-1.5B-Instruct wrapper
│   └── vlm.py                      # Composite VLM model
├── data/
│   ├── image_processing.py         # CLIP image transforms
│   ├── conversation.py             # Conversation tokenization + label masking
│   ├── dataset.py                  # LLaVAPretrainDataset
│   └── collator.py                 # Batch collation
└── training/
    ├── lr_scheduler.py             # Cosine warmup scheduler
    ├── checkpoint.py               # Checkpoint save/load
    └── trainer.py                  # Training loop
```

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-3 | High (connector only), follows LLaVA |
| Weight Decay | 0.0 | No regularization needed for small connector |
| Batch Size | 256 (effective) | Via 8 × 32 (batch × accumulation) |
| Warmup | 3% of steps | Short warmup; model components are pre-trained |
| Schedule | Cosine decay | Smooth convergence to near-zero lr |
| Precision | bf16 | Mixed precision for efficiency |
| Epochs | 1 | Single pass over 558K samples (if using LLaVA-Pretrain) |

## Performance & Memory

**Trainable Parameters**: ~3.9M (only the MLP connector)
**Frozen Parameters**: ~1.8B (CLIP + LLM)
**GPU Memory**: ~6–8 GB (with batch_size=8, bf16 precision)

Expected training time on a single RTX 3060 (12GB) for 558K samples:
- ~3 days at batch_size=8 with gradient_accumulation=32

## What Gets Trained

Only the connector's weights are updated:
- `model.connector.mlp[0].weight` — (1536, 1024)
- `model.connector.mlp[0].bias` — (1536,)
- `model.connector.mlp[2].weight` — (1536, 1536)
- `model.connector.mlp[2].bias` — (1536,)

Both the vision encoder (`vision_encoder.model`) and LLM (`language_model.model`) are frozen via `requires_grad=False`.

## Data Format Details

The dataset expects a JSON file with the LLaVA-Pretrain structure:
- Each entry must have `"image"` (relative path to image), `"conversations"` (list of turns)
- Conversations are `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`
- The `<image>` placeholder in the human message is replaced with visual token embeddings during training
- Labels are masked (`-100`) for all tokens except the assistant's response

## Inference Details

During inference:
1. The image is preprocessed via CLIP's image processor (224×224 resize, normalization)
2. CLIP encodes it to 256 patch tokens (1024-dim each)
3. The connector projects to 1536-dim (LLM embedding space)
4. Text prompt is tokenized and embedded via the LLM's embedding layer
5. Visual and text embeddings are concatenated and fed to the LLM
6. Autoregressive generation proceeds with `model.generate(...)`

The `<image>` token in the prompt is automatically replaced with visual embeddings; it does not appear in the final token sequence.

## Next Steps

This implementation covers **Stage 1: Feature Alignment**. Future extensions might include:

1. **Stage 2: Full Model Tuning** — Unfreeze LLM layers and fine-tune on instruction-following data
2. **Cross-Attention Connector** — Replace MLP with a learned cross-attention mechanism for better spatial reasoning
3. **Higher-Resolution Images** — Support variable image resolutions and dynamic patching
4. **Multi-Image Support** — Handle multiple images per prompt
5. **Evaluation** — Add benchmarks (VQA, captioning, visual reasoning tasks)

## Troubleshooting

**Out of Memory (OOM)**
- Reduce `per_device_batch_size` (try 4 or 2)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `bf16` mixed precision (already default)

**Slow Data Loading**
- Increase `dataloader_num_workers` (try 8 or 16)
- Pre-extract CLIP features to disk to bypass image I/O

**NaN Loss**
- Check label masking: visual token positions should have `labels = -100`
- Verify `attention_mask` has correct shape and no all-zero rows
- Reduce learning rate if instability persists

## References

- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **CLIP**: [Learning Transferable Models for Compositional Vision](https://arxiv.org/abs/2103.14030)
- **Qwen**: [Qwen2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)

## License

MIT License. See LICENSE file (if present) for details.

## Contributing

Contributions are welcome. Please:
1. Test your changes with a small dataset subset
2. Verify trainable params count and gradient flow
3. Include clear commit messages

## Questions?

For issues or questions about the implementation, check the inline comments in `vlm_model/vlm.py` (especially `prepare_inputs_embeds()`) and `training/trainer.py` for detailed logic.
