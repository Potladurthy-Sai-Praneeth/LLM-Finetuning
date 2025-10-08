"""
Inference utilities and model comparison for base vs finetuned models
"""

import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
import argparse
from pathlib import Path
import yaml


def load_config():
    """Load configuration from YAML file"""
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def get_merged_model(base_model_id: str, adapter_path: str, output_path: str):
    """
    Merge PEFT adapter with base model and save
    
    Args:
        base_model_id: Base model identifier
        adapter_path: Path to trained adapter
        output_path: Path to save merged model
    
    Returns:
        Merged model
    """
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Also save the processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    return merged_model


def load_model_for_inference(model_path: str, is_base_model: bool = False):
    """
    Load model for inference
    
    Args:
        model_path: Path to the model or model ID
        is_base_model: Whether this is a base model ID (from HuggingFace) or local path
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading {'base' if is_base_model else 'finetuned'} model from: {model_path}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"✓ {'Base' if is_base_model else 'Finetuned'} model loaded successfully")
    return model, processor


def generate_response(model, processor, image, text_prompt: str, max_new_tokens: int = 512):
    """
    Generate response for image-text input
    
    Args:
        model: Trained model
        processor: Model processor
        image: PIL Image or None
        text_prompt: Text prompt
        max_new_tokens: Maximum generation length
    
    Returns:
        Generated text
    """
    # Format the prompt as a chat message (following parallel_finetune.py pattern)
    system_message = "You are an expert in Medical Diagnosis and treatment. You are given a image. You need to provide a description of the disease or anamoly or sometimes suggest possible treatments analysing the image."
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
            ],
        },
    ]
    
    # Add image if provided
    if image is not None:
        messages[1]["content"].insert(1, {"type": "image", "image": image})
    
    # Apply chat template
    chat_text = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Process inputs
    inputs = processor(
        text=chat_text,
        images=[image] if image is not None else None,
        return_tensors="pt",
        padding=True
    )
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response


def compare_models(image_path: str, text_prompt: str, config: dict):
    """
    Compare base model and finetuned model outputs
    
    Args:
        image_path: Path to input image (optional)
        text_prompt: User's text prompt
        config: Configuration dictionary
    """
    # Load image if provided
    image = None
    if image_path and os.path.exists(image_path):
        print(f"\nLoading image from: {image_path}")
        image = Image.open(image_path)
        print(f"✓ Image loaded successfully (size: {image.size})")
    elif image_path:
        print(f"Warning: Image path provided but file not found: {image_path}")
        print("Proceeding with text-only inference...")
    
    # Get model paths from config
    base_model_id = config['model']['BASE_MODEL_ID']
    finetuned_model_path = os.path.join(
        config['training']['OUTPUT_DIR'], 
        "final_merged_model"
    )
    
    # Check if finetuned model exists
    if not os.path.exists(finetuned_model_path):
        print(f"\nWarning: Finetuned model not found at {finetuned_model_path}")
        print("Please train the model first using parallel_finetune.py")
        print("Running inference on base model only...\n")
        
        # Load only base model
        base_model, base_processor = load_model_for_inference(base_model_id, is_base_model=True)
        
        print("\n" + "="*80)
        print("BASE MODEL OUTPUT")
        print("="*80)
        base_response = generate_response(base_model, base_processor, image, text_prompt)
        print(base_response)
        print("="*80)
        return
    
    # Load both models
    print("\n[STEP 1] Loading base model...")
    base_model, base_processor = load_model_for_inference(base_model_id, is_base_model=True)
    
    print("\n[STEP 2] Loading finetuned model...")
    finetuned_model, finetuned_processor = load_model_for_inference(finetuned_model_path, is_base_model=False)
    
    # Generate responses
    print("\n[STEP 3] Generating base model response...")
    base_response = generate_response(base_model, base_processor, image, text_prompt)
    
    print("\n[STEP 4] Generating finetuned model response...")
    finetuned_response = generate_response(finetuned_model, finetuned_processor, image, text_prompt)
    
    # Print comparison
    print("\n" + "="*80)
    print("INFERENCE COMPARISON")
    print("="*80)
    print(f"\nUser Prompt: {text_prompt}")
    if image_path:
        print(f"Image: {image_path}")
    print("\n" + "-"*80)
    print("BASE MODEL OUTPUT:")
    print("-"*80)
    print(base_response)
    print("\n" + "-"*80)
    print("FINETUNED MODEL OUTPUT:")
    print("-"*80)
    print(finetuned_response)
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare base and finetuned Gemma-3 VLM model outputs"
    )
    parser.add_argument(
        '--image_path', 
        type=str, 
        default=None,
        help="Path to input image (optional)"
    )
    parser.add_argument(
        '--text_prompt', 
        type=str, 
        required=True, 
        help="Text prompt for generation"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Run comparison
    compare_models(args.image_path, args.text_prompt, config)