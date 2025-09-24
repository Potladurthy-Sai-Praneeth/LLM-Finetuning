"""
Inference utilities and model merging functions
"""

import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import argparse


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
        torch_dtype=torch.bfloat16,
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


def load_model_for_inference(model_path: str):
    """
    Load model for inference
    
    Args:
        model_path: Path to the model
    
    Returns:
        Tuple of (model, processor)
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    return model, processor


def generate_response(model, processor, image, text_prompt: str, max_length: int = 256):
    """
    Generate response for image-text input
    
    Args:
        model: Trained model
        processor: Model processor
        image: Input image
        text_prompt: Text prompt
        max_length: Maximum generation length
    
    Returns:
        Generated text
    """
    inputs = processor(
        text=text_prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PEFT adapter with base model for inference")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--text_prompt', type=str, required=True, help="Text prompt for generation")

    args = parser.parse_args()

    model , processor = load_model_for_inference(args.output_path)
    response = generate_response(model, processor, args.image_path, args.text_prompt)
    print(f'--'*20)
    print(f'User Prompt: \n {args.text_prompt}')
    print(f'--'*20)
    print('\n')
    print("Generated Response: \n", response)    
    print(f'--'*20)