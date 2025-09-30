# LLM Fine-tuning with QLoRA + FSDP

Fine-tuning Gemma-3 Vision-Language Model (VLM) for medical image diagnosis using QLoRA (Quantized Low-Rank Adaptation) with FSDP (Fully Sharded Data Parallel) for multi-GPU training/finetuning.


## 🎯 Overview

This project implements efficient fine-tuning of large vision-language models using:
- **QLoRA**: 4-bit quantization with Low-Rank Adaptation for memory-efficient training
- **FSDP**: Fully Sharded Data Parallel for distributed multi-GPU training
- **Gemma-3 VLM**: Google's Gemma-3-4B model with vision capabilities (Siglip vision tower)
- **Medical Domain**: Fine-tuned on MedPix-Grouped-QA dataset for medical image diagnosis

---

## ✨ Features

- **Multi-GPU Training**: FSDP support for distributed training across multiple GPUs
- **Memory Efficient**: QLoRA with 4-bit NF4 quantization reduces memory footprint by ~75%
- **Vision-Language Support**: Handles both image and text inputs for multimodal tasks
- **Configurable**: YAML-based configuration for easy hyperparameter tuning
- **Model Comparison**: Inference script compares base vs fine-tuned model outputs
- **Medical Domain**: Pre-configured for medical image diagnosis (customizable for other domains)

---
## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LLM-Finetuning.git
cd LLM-Finetuning
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `transformers`: Model loading and training
- `peft`: Parameter-Efficient Fine-Tuning (LoRA)
- `accelerate`: FSDP and distributed training
- `trl`: SFTTrainer for supervised fine-tuning
- `bitsandbytes`: 4-bit quantization
- `datasets`: Dataset loading
- `torch`: PyTorch deep learning framework
- `deepspeed`: (Optional) Alternative to FSDP

---
## 🎓 Usage

### Training

**Distributed Setup (Multiple GPUs):**
```bash
torchrun --nproc_per_node=2  parallel_finetune.py
```


### Single-GPU Training
```bash
python finetuning.py
```

### Inference


**With Image:**
```bash
python inference.py \
  --image_path "path/to/medical_scan.jpg" \
  --text_prompt "What condition is shown in this X-ray?"
```

**Text-Only:**
```bash
python inference.py \
  --text_prompt "Describe the symptoms of pneumonia in chest X-rays"
```

---
## ⚠️ Known Issues with FSDP + QLoRA

Combining **FSDP** (Fully Sharded Data Parallel) with **QLoRA** (Quantized LoRA) presents unique challenges due to the interaction between distributed training, quantization, and mixed precision. Below are the most common issues and their explanations:

### 1. **Mixed Precision Conflicts**

**Problem:**
- FSDP expects uniform precision across all parameters (e.g., all `bfloat16`).
- QLoRA uses 4-bit quantized weights (via `bitsandbytes`) for the base model, but LoRA adapters are in `float32` or `bfloat16`.
- This creates a **precision mismatch** when FSDP tries to shard and synchronize parameters.

**Root Cause:**
- FSDP's `fsdp_use_orig_params=True` requires all parameters to be wrapped consistently, but quantized layers (`Linear4bit`) don't behave like standard `nn.Linear`.
- Gradient checkpointing (recomputing activations) can cause dtype mismatches when recomputing 4-bit layers.

**Workaround:**
```python
for name, param in trainer.model.named_parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)
```
This ensures all trainable params (LoRA adapters) are in `bfloat16` before FSDP wrapping.

---

### 2. **CPU RAM Exhaustion During Initialization**

**Problem:**
- Each FSDP rank (process) initially loads the **full model** into CPU RAM before sharding.
- With QLoRA, the full model is ~20-30GB (pre-quantization weights + vision tower), so 2 GPUs = 2 processes = 40-60GB CPU RAM spike.
- Even with `low_cpu_mem_usage=True`, it is still memory-intensive.

**Root Cause:**
- Pre-loading the model in `load_model_and_processor()` bypasses FSDP's `fsdp_cpu_ram_efficient_loading`, which only works when the model is loaded **inside** the trainer.
- `bitsandbytes` quantization requires loading raw weights to CPU, quantizing, then moving to GPU—this temporarily doubles memory.

---

### 3. **GPU OOM (Out of Memory) During Training**

**Problem:**
- Even with 4-bit quantization and FSDP sharding, **activations** (intermediate tensors) dominate GPU memory.
- Vision-Language Models (VLMs) like Gemma-3 process high-resolution images, creating massive activation tensors in the vision tower and multimodal projector.

---

### 4. **Communication Overhead and Slow Training**

**Problem:**
- FSDP requires frequent all-gather/reduce-scatter operations to synchronize sharded parameters across GPUs.
- With QLoRA, **activations are NOT sharded** (only params/grads), so each GPU still computes full forward passes—limiting speedup.
- Vision models have large activation tensors, increasing inter-GPU communication time.

---

## 📁 Project Structure

```
LLM-Finetuning/
├── config.yaml                  # Configuration file (models, training params, FSDP settings)
├── data_preprocessing.py        # Custom dataset and collate function
├── parallel_finetune.py         # Main training script with FSDP
├── deepspeed_finetune.py        # Alternative training with DeepSpeed
├── finetuning.py                # Single-GPU training script
├── inference.py                 # Inference and model comparison
├── requirements.txt             # Python dependencies
├── ds_config.json              # DeepSpeed configuration
└── README.md                    # This file
```
--- 