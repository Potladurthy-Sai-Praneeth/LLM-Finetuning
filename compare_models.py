"""
Model Comparison Script: Gemma 3 Base vs Fine-tuned with LoRA Adapters

This script compares the outputs of the Gemma 3 model before and after fine-tuning
by loading the base model, merging LoRA adapters, and running test queries.
"""

import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
from datasets import load_dataset
from pathlib import Path
import argparse
from datetime import datetime
import yaml


class ModelComparator:
    """Compare base and fine-tuned models on multimodal tasks"""
    
    def __init__(self, base_model_id: str, chat_model_id: str, adapter_path: str, config: dict):
        """
        Initialize the comparator
        
        Args:
            base_model_id: HuggingFace model ID for base model
            chat_model_id: HuggingFace model ID for chat model
            adapter_path: Path to trained LoRA adapter
            config: Configuration dictionary
        """
        self.base_model_id = base_model_id
        self.chat_model_id = chat_model_id
        self.adapter_path = adapter_path
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Base Model: {base_model_id}")
        print(f"Chat Model: {chat_model_id}")
        print(f"Adapter Path: {adapter_path}")
        
    def load_base_model(self):
        """Load the base Gemma 3 model"""
        print("\n" + "="*80)
        print("Loading Base Model...")
        print("="*80)
        
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.base_processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            trust_remote_code=True
        )
        
        print("✓ Base model loaded successfully")
        
    def load_finetuned_model(self):
        """Load and merge the LoRA adapter with base model"""
        print("\n" + "="*80)
        print("Loading and Merging Fine-tuned Adapter...")
        print("="*80)
        
        # Load base model for merging
        print(f"Loading base model: {self.base_model_id}")
        base_model_for_merge = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load adapter
        print(f"Loading adapter from: {self.adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(
            base_model_for_merge, 
            self.adapter_path
        )
        
        # Merge and unload
        print("Merging adapter with base model...")
        self.finetuned_model = model_with_adapter.merge_and_unload()
        
        # Use same processor
        self.finetuned_processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            trust_remote_code=True
        )
        
        print("✓ Fine-tuned model ready")
        
    def generate_response(self, model, processor, image, text_prompt: str, 
                         max_new_tokens: int = 512, temperature: float = 0.7):
        """
        Generate response for image-text input
        
        Args:
            model: Model to use for generation
            processor: Processor for the model
            image: PIL Image
            text_prompt: Text question/prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # System message for medical domain
        system_message = (
            "You are an expert in Medical Diagnosis and treatment. "
            "You are given a image. You need to provide a description of "
            "the disease or anamoly or sometimes suggest possible treatments "
            "analysing the image."
        )
        
        # Format as chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        
        # Apply chat template
        chat_text = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Process inputs
        inputs = processor(
            text=chat_text,
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id 
                    if processor.tokenizer.pad_token_id is not None 
                    else processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        # The response typically includes the full conversation
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        elif "assistant" in response:
            response = response.split("assistant")[-1].strip()
            
        return response
    
    def load_test_samples(self, num_samples: int = 5):
        """
        Load test samples from the dataset
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of test samples
        """
        print("\n" + "="*80)
        print(f"Loading Test Samples from Dataset...")
        print("="*80)
        
        dataset_id = self.config['dataset']['DATASET_ID']
        img_size = self.config['model']['IMG_SIZE']
        
        print(f"Dataset: {dataset_id}")
        
        # Load dataset
        dataset = load_dataset(dataset_id, split="train")
        
        # Get samples
        test_samples = []
        sample_count = 0
        
        for idx, sample in enumerate(dataset):
            if sample_count >= num_samples:
                break
                
            # Get image
            image = sample['image']
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                continue
                
            # Resize image
            image = image.resize((img_size, img_size))
            
            # Get questions and answers
            questions = sample.get('questions', [])
            answers = sample.get('answers', [])
            
            if not questions or not answers:
                continue
            
            # Take first question-answer pair from this sample
            test_samples.append({
                'image': image,
                'question': questions[0],
                'ground_truth': answers[0],
                'sample_id': idx
            })
            
            sample_count += 1
            
        print(f"✓ Loaded {len(test_samples)} test samples")
        return test_samples
    
    def compare_on_samples(self, test_samples: list, output_file: str = None):
        """
        Compare base and fine-tuned models on test samples
        
        Args:
            test_samples: List of test samples
            output_file: Path to save comparison results
            
        Returns:
            List of comparison results
        """
        print("\n" + "="*80)
        print("Running Model Comparison...")
        print("="*80)
        
        results = []
        
        for idx, sample in enumerate(test_samples):
            print(f"\n--- Sample {idx + 1}/{len(test_samples)} ---")
            print(f"Question: {sample['question'][:100]}...")
            
            # Generate response from base model
            print("Generating base model response...")
            base_response = self.generate_response(
                self.base_model,
                self.base_processor,
                sample['image'],
                sample['question']
            )
            
            # Generate response from fine-tuned model
            print("Generating fine-tuned model response...")
            finetuned_response = self.generate_response(
                self.finetuned_model,
                self.finetuned_processor,
                sample['image'],
                sample['question']
            )
            
            # Store results
            result = {
                'sample_id': sample['sample_id'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'base_model_response': base_response,
                'finetuned_model_response': finetuned_response,
                'image_size': sample['image'].size
            }
            
            results.append(result)
            
            # Print comparison
            print(f"\nGround Truth: {sample['ground_truth'][:200]}...")
            print(f"\nBase Model: {base_response[:200]}...")
            print(f"\nFine-tuned Model: {finetuned_response[:200]}...")
            print("-" * 80)
            
        # Save results
        if output_file:
            self.save_results(results, output_file)
            
        return results
    
    def save_results(self, results: list, output_file: str):
        """
        Save comparison results to JSON file
        
        Args:
            results: List of comparison results
            output_file: Path to output file
        """
        print(f"\nSaving results to: {output_file}")
        
        # Prepare data for JSON (remove PIL images)
        json_results = []
        for result in results:
            json_result = {k: v for k, v in result.items() if k != 'image'}
            json_results.append(json_result)
        
        # Add metadata
        output_data = {
            'metadata': {
                'base_model': self.base_model_id,
                'adapter_path': self.adapter_path,
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(results)
            },
            'results': json_results
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Results saved successfully")
        
        # Also save a human-readable version
        txt_file = output_file.replace('.json', '.txt')
        self.save_readable_results(results, txt_file)
        
    def save_readable_results(self, results: list, output_file: str):
        """
        Save results in human-readable format
        
        Args:
            results: List of comparison results
            output_file: Path to output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GEMMA 3 MODEL COMPARISON: BASE vs FINE-TUNED\n")
            f.write("="*80 + "\n\n")
            f.write(f"Base Model: {self.base_model_id}\n")
            f.write(f"Adapter Path: {self.adapter_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Number of Samples: {len(results)}\n")
            f.write("="*80 + "\n\n")
            
            for idx, result in enumerate(results):
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {idx + 1}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Sample ID: {result['sample_id']}\n")
                f.write(f"Image Size: {result['image_size']}\n\n")
                
                f.write(f"QUESTION:\n{result['question']}\n\n")
                
                f.write(f"GROUND TRUTH:\n{result['ground_truth']}\n\n")
                
                f.write(f"BASE MODEL RESPONSE:\n{result['base_model_response']}\n\n")
                
                f.write(f"FINE-TUNED MODEL RESPONSE:\n{result['finetuned_model_response']}\n\n")
                
                f.write("-"*80 + "\n")
                
        print(f"✓ Readable results saved to: {output_file}")


def load_config(config_path: str = None):
    """Load configuration from YAML file"""
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir / "finetune" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def main():
    """Main function to run model comparison"""
    parser = argparse.ArgumentParser(
        description="Compare Gemma 3 base model vs fine-tuned model with LoRA adapters"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="final_adapter",
        help="Path to the trained LoRA adapter weights"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Base model ID (default: from config.yaml)"
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Chat model ID (default: from config.yaml)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of test samples to evaluate (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.json",
        help="Output file path for results (default: comparison_results.json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Get base model from config if not provided
    base_model_id = args.base_model if args.base_model else config['model']['BASE_MODEL_ID']
    
    # Initialize comparator
    print("\n" + "="*80)
    print("GEMMA 3 MODEL COMPARISON")
    print("="*80)
    comparator = ModelComparator(base_model_id, args.chat_model, args.adapter_path, config)
    
    # Load models
    comparator.load_base_model()
    comparator.load_finetuned_model()
    
    # Load test samples
    test_samples = comparator.load_test_samples(num_samples=args.num_samples)
    
    # Run comparison
    results = comparator.compare_on_samples(test_samples, output_file=args.output)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Total samples evaluated: {len(results)}")
    print(f"Results saved to: {args.output}")
    print(f"Readable version: {args.output.replace('.json', '.txt')}")
    

if __name__ == "__main__":
    main()
