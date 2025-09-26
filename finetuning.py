"""
Fine-tuning Script for Vision-Language Models
"""

import os
from collections import Counter
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_preprocessing import CustomDataset
from inference import get_merged_model


class Trainer:
    """Handles single GPU training setup and execution"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        # Check for environment variable first
        config_path = os.getenv('CONFIG_PATH')
        
        if config_path:
            config_path = Path(config_path)
        else:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.yaml"
        
        if not config_path.exists():
            # Try additional common locations
            possible_paths = [
                Path.cwd() / "config.yaml",
                Path("/kaggle/working/config.yaml"),
                Path("/content/config.yaml"),
                Path("./config.yaml")
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError(f"Config file not found. Tried: {config_path} and {possible_paths}")
        
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Set up environment variables
        env_vars = self.config.get('environment', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
    
    def _get_quantization_config(self):
        """Get quantization configuration for 4-bit training"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    
    def _get_peft_config(self):
        """Get PEFT (LoRA) configuration - optimized for memory efficiency"""
        return LoraConfig(
            lora_alpha=self.config['lora']['LORA_ALPHA'],
            lora_dropout=self.config['lora']['LORA_DROPOUT'],
            r=self.config['lora']['LORA_R'],
            bias="none",
            task_type=self.config['lora']['TASK_TYPE'],
            modules_to_save=self.config['lora']['MODULES_TO_SAVE'],
            target_modules=self.config['lora']['TARGET_MODULES']
        )
    
    def load_model_and_processor(self):
        """Load and configure model and processor for single GPU training"""
        quantization_config = self._get_quantization_config()

        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['BASE_MODEL_ID'],
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False

        # Load processor
        processor = AutoProcessor.from_pretrained(
            self.config['model']['CHAT_MODEL_ID'],
            trust_remote_code=True
        )

        # Apply PEFT
        peft_config = self._get_peft_config()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        # Freeze all parameters except LoRA layers
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_A." in name or ".lora_B." in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        return model, processor, peft_config
    
    def collate_fn(self, batch):
        if self.processor is None:
            raise ValueError("Processor not initialized")

        texts = []
        images = []
        for example in batch:
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(example['messages'][1]['content'][1]['image'])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True, max_length=self.config['model']['MAX_SEQ_LENGTH'], truncation=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        
        return batch
    
    def get_training_args(self):
        """Get training arguments configuration for single GPU training"""
        effective_batch_size = min(self.config['training']['BATCH_SIZE'], 1) 
        gradient_accumulation_steps = self.config['training']['GRADIENT_ACCUMULATION_STEPS']

        return SFTConfig(
            output_dir=self.config['training']['OUTPUT_DIR'],
            num_train_epochs=self.config['training']['NUM_TRAIN_EPOCHS'],
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=self.config['training']['LOGGING_STEPS'],
            save_strategy="epoch",
            learning_rate=self.config['training']['LEARNING_RATE'],
            bf16=True,
            lr_scheduler_type="cosine",
            dataset_text_field='',
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            save_only_model=True,
        )
    
    def train(self):
        """Main training function for single GPU"""
        try:
            print("Starting single GPU training")
            print(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            print(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            
            # Load model, processor, and configuration
            model, processor, peft_config = self.load_model_and_processor()
            self.model = model
            self.processor = processor
            training_args = self.get_training_args()

            # Load dataset
            dataset = CustomDataset(load_dataset(self.config['dataset']['DATASET_ID'], split="train"))

            # Initialize trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                processing_class=processor,
                data_collator=self.collate_fn,
            )
            
            # Start training
            trainer.train()

            # Save model and merge
            print("Training completed. Saving model...")
            trainer.save_model()

            print("Merging model...")
            merged_model = get_merged_model(
                self.config['model']['BASE_MODEL_ID'],
                self.config['training']['OUTPUT_DIR'],
                self.config['training']['MERGED_MODEL_DIR']
            )
            print(f"Model saved to: {self.config['training']['OUTPUT_DIR']}")
            print(f"Merged model saved to: {self.config['training']['MERGED_MODEL_DIR']}")
                
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise
    
    def run(self):
        """Run training on single GPU"""
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")
        
        print("Starting single GPU training...")
        self.train()


def main():
    """Main entry point"""
    try:
        trainer = Trainer()
        trainer.run()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()