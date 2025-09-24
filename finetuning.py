"""
Distributed Fine-tuning Script for Vision-Language Models
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig, 
    AutoProcessor, 
    AutoModelForImageTextToText, 
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_preprocessing import collate_fn,process_dataset
from inference import get_merged_model


class DistributedTrainer:
    """Handles distributed training setup and execution"""
    
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
        
        # Validate required configs
        if not self.base_model_id:
            raise ValueError("BASE_MODEL_ID must be set in environment variables")
        if not self.dataset_id:
            raise ValueError("DATASET_ID must be set in environment variables")
    
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
        """Get PEFT (LoRA) configuration"""
        return LoraConfig(
            lora_alpha=32, 
            lora_dropout=0.1, 
            r=32, 
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=[
                "lm_head",
                "embed_tokens",
            ],
            target_modules='all-linear'
        )
    
    def setup_ddp(self, rank, world_size):
        """Initialize distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    def cleanup_ddp(self):
        """Clean up distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def collate_fn(self,batch):
        texts = []
        images = []
        for example in batch:
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(example['messages'][1]['content'][1]['image'])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        batch['labels'] = batch["input_ids"].clone()
        
        return batch
    
    def load_model_and_processor(self, rank, world_size):
        """Load and configure model and processor"""
        quantization_config = self._get_quantization_config()
        device_map = {"": rank} if world_size > 1 else "auto"
        
        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            trust_remote_code=True
        )
        
        # Apply PEFT
        peft_config = self._get_peft_config()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
        return model, processor, peft_config
    
    def get_training_args(self, rank, world_size):
        """Get training arguments configuration"""
        is_main_process = rank == 0
        
        return SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=5,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10 if is_main_process else 1000,
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            bf16=True,
            lr_scheduler_type="cosine",
            dataset_text_field='',
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            # DDP specific configurations
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            local_rank=rank,
            report_to=[] if not is_main_process else None,
            # Save only on main process
            save_only_model=True,
        )
    
    def train_worker(self, rank, world_size):
        """Main training worker function"""
        try:
            # Setup distributed training
            if world_size > 1:
                self.setup_ddp(rank, world_size)
            
            is_main_process = rank == 0
            
            if is_main_process:
                print(f"Starting training on {world_size} GPU(s)")
                print(f"Model: {self.base_model_id}")
                print(f"Dataset: {self.dataset_id}")
            
            # Load model, processor, and configuration
            model, processor, peft_config = self.load_model_and_processor(rank, world_size)
            self.model = model
            self.processor = processor
            training_args = self.get_training_args(rank, world_size)
            
            # Load dataset
            dataset = load_dataset(self.dataset_id, split="train")
            dataset = dataset.select(range(self.dataset_size)) 
            dataset = process_dataset(dataset)
            
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
            
            # Save model and merge (only on main process)
            if is_main_process:
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
            
            # Synchronize all processes
            if world_size > 1:
                dist.barrier()
                
        except Exception as e:
            print(f"Error in training worker {rank}: {str(e)}")
            raise
        finally:
            # Cleanup
            if world_size > 1:
                self.cleanup_ddp()
    
    def run(self):
        """Run training with appropriate configuration"""
        world_size = torch.cuda.device_count()
        
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
        
        print(f"Available GPUs: {world_size}")
        
        if world_size > 1:
            print("Starting distributed training...")
            mp.spawn(
                self.train_worker, 
                args=(world_size,), 
                nprocs=world_size, 
                join=True
            )
        else:
            print("Starting single GPU training...")
            self.train_worker(0, 1)


def main():
    """Main entry point"""
    try:
        trainer = DistributedTrainer()
        trainer.run()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()