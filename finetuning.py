"""
Fine-tuning Script for Vision-Language Models
"""

import os
from collections import Counter
import torch
import torch.nn as nn
from dotenv import load_dotenv
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
        load_dotenv()
        self.processor = None
        self.model = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        self.base_model_id = os.getenv("BASE_MODEL_ID")
        self.batch_size = int(os.getenv("BATCH_SIZE", 1))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", 3))
        self.learning_rate = float(os.getenv("LEARNING_RATE", 2e-4))
        self.dataset_id = os.getenv("DATASET_ID")
        self.output_dir = os.getenv("OUTPUT_DIR", "gemma-medical")
        self.merged_model_dir = os.getenv("MERGED_MODEL_DIR", f"{self.output_dir}-merged")
        self.dataset_size = int(os.getenv("DATASET_SIZE", 1000))
        self.chat_model_id = os.getenv("CHAT_MODEL_ID")
        
        # Validate required configs
        if not self.base_model_id:
            raise ValueError("BASE_MODEL_ID must be set in environment variables")
        if not self.dataset_id:
            raise ValueError("DATASET_ID must be set in environment variables")
        if not self.chat_model_id:
            raise ValueError("CHAT_MODEL_ID must be set in environment variables")
    
    
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
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,  
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=[
                "lm_head",
                "embed_tokens",
            ],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  
                "gate_proj", "up_proj", "down_proj",    
            ]
        )
    
    def load_model_and_processor(self):
        """Load and configure model and processor for single GPU training"""
        quantization_config = self._get_quantization_config()

        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False

        # Load processor
        processor = AutoProcessor.from_pretrained(
            self.chat_model_id,
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

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True, max_length=512, truncation=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        

        return batch
    
    def get_training_args(self):
        """Get training arguments configuration for single GPU training"""
        effective_batch_size = min(self.batch_size, 1) 
        gradient_accumulation_steps = 1

        return SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=self.learning_rate,
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
            print(f"Model: {self.base_model_id}")
            print(f"Dataset: {self.dataset_id}")
            
            # Load model, processor, and configuration
            model, processor, peft_config = self.load_model_and_processor()
            self.model = model
            self.processor = processor
            training_args = self.get_training_args()

            # Load dataset
            dataset = CustomDataset(load_dataset(self.dataset_id, split="train"))

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
                self.base_model_id,
                self.output_dir,
                self.merged_model_dir
            )
            print(f"Model saved to: {self.output_dir}")
            print(f"Merged model saved to: {self.merged_model_dir}")
                
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