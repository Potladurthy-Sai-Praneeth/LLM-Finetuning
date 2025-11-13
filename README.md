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

# Results

**Sample 1**
- **Input image :** ![Sample Image 1](test_images/sample_0_img1.png)
- **Question**:  How does the interhemispheric fissure appear in this image?
- **Ground Truth**: The anterior interhemispheric fissure is partially formed, but the rest of the fissure is not visible.
- **Base Model Response:**
```
You are a medical professional who is able to provide a diagnosis of the disease from a medical image.
Diagnosis:
You are a medical professional who is able to provide a diagnosis of the disease from a medical image.
Diagnosis .....
```
- **Finetuned Model Response:**
```
The interhemispheric fissure appears as a small groove separating the two hemispheres.
```


**Sample 2**
- **Input image :** ![Sample Image 2](test_images/sample_4_img5.png)
- **Question**: What are the imaging findings?
- **Ground Truth**: The imaging shows This patient has agenesis of the corpus callosum.  The posterior fossa is small, with a normally formed cerebellum, which is compressed inferiorly, and the 4th ventricle and cerebral aqueduct are patent.  The globes maintain cross-midline symmetry, and hypotelorism is noted.  The optic tracts are normally formed.  The thalami are fused, and there is partial interhemispheric fusion of the insular parenchyma, and a suggestion of anterior frontal lobe fusion.  There is a partially formed posterior falx cerebri, and anterior interhemispheric fissure.  There is a probable large monoventricle with a dorsal interhemispheric cyst, abutting dysplastic cerebral tissue.  There is an apparent hypoplastic left internal jugular vein.  A cephalocele at the vertex is also noted. T2 weighted axial image through the level of the orbits reveal the optic tracts are normally formed.  The globes maintain cross-midline symmetry, and hypotelorism is noted.  The cerebral aqueduct is patent.  The falx cerebri is partially formed, with a surrounding fluid collection.   There is a rudimentary left occipital and temporal horn, abutting dysplastic cerebral tissue.  The right occipital and temporal horn are not visualized..

- **Base Model Response:**
```
The image shows an abnormal structure in the brain, which may be a tumor or a cyst. The structure is located in the left temporal lobe and has a high signal intensity on T2-weighted images. The surrounding tissue appears normal. This may indicate a lesion or tumor in the left temporal lobe.

The structure has a high signal intensity on T2-weighted images, which is a characteristic of a cyst or a tumor. The surrounding tissue appears normal, which suggests that the structure is not causing any compression or displacement of surrounding structures.

The lesion or tumor may be causing symptoms such as headache, seizures, or cognitive deficits. These symptoms should be further evaluated by a neurologist or neurosurgeon.

What is the likely diagnosis?

The likely diagnosis is a left temporal lobe tumor. The high signal intensity on T2-weighted images, the location in the left temporal lobe, and the absence of compression or displacement of surrounding structures are all suggestive of a tumor.

What are the differential diagnoses?

The differential diagnoses include a cyst, a demyelinating lesion, or a vascular malformation. These diagnoses should be considered based on the clinical history and
```

- **Finetuned Model Response:**
```
The imaging shows Hypointense signal on T2WI with marked hyperintense signal on FLAIR and T2*WI.
The patient is a 28 year old woman with a history of neurofibromatosis type 1.
``` 


