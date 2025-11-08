import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import yaml

from data_preprocessing import CustomDataset


def setup_distributed():
    """
    Initialize distributed training environment.
    This function detects if running in a distributed setting and initializes accordingly.
    Works with torchrun, Vertex AI, and other distributed launchers.
    """
    # Check if already initialized
    if dist.is_available() and dist.is_initialized():
        print("Distributed training already initialized")
        return True
    
    # Detect distributed environment variables
    # These can be set by torchrun, Vertex AI, or other launchers
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Check if we're in a distributed setting
    if world_size > 1:
        print(f"Initializing distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        
        # Initialize the process group
        if not dist.is_initialized():
            # Set backend
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            
            # Initialize process group
            dist.init_process_group(
                backend=backend,
                init_method='env://',  # Use environment variables
                world_size=world_size,
                rank=rank
            )
            
            # Set the device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            print(f"✓ Distributed training initialized on rank {rank}/{world_size}")
        
        return True
    else:
        print("Running in single GPU/CPU mode")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False


def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print("✓ Distributed training cleaned up")


class DistributedTrainer:
    """Handles distributed training setup and execution for Vertex AI"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.config = {}
        self.is_distributed = False
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        
        self._load_config()
        
    def is_main_process(self):
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
    
    def print_rank0(self, *args, **kwargs):
        """Print only from the main process"""
        if self.is_main_process():
            print(*args, **kwargs)

    def _load_config(self):
        """Load configuration from YAML file"""
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
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
        """Get PEFT (LoRA) configuration"""
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
        """Load model and processor"""
        self.print_rank0(f"Loading model: {self.config['model']['BASE_MODEL_ID']}")
        
        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['BASE_MODEL_ID'],
            quantization_config=self._get_quantization_config(),
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False
        self.print_rank0("✓ Model loaded successfully")

        # Freeze base model parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        self.print_rank0(f"Loading processor: {self.config['model']['CHAT_MODEL_ID']}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config['model']['CHAT_MODEL_ID'],
            trust_remote_code=True,
            use_fast=True
        )
        self.print_rank0("✓ Processor loaded successfully")

        self.print_rank0("Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(model)
        self.print_rank0("✓ Model configuration completed")

    def get_training_args(self):
        """Get training arguments configuration for distributed training"""
        # Batch size per device (already set in config)
        per_device_batch_size = int(self.config['training']['BATCH_SIZE'])
        gradient_accumulation_steps = int(self.config['training']['GRADIENT_ACCUMULATION_STEPS'])
        
        # Calculate effective batch size
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps * self.world_size
        
        self.print_rank0(f"Training configuration:")
        self.print_rank0(f"  - World size (GPUs): {self.world_size}")
        self.print_rank0(f"  - Per-device batch size: {per_device_batch_size}")
        self.print_rank0(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        self.print_rank0(f"  - Effective batch size: {effective_batch_size}")

        return SFTConfig(
            output_dir=self.config['training']['OUTPUT_DIR'],
            num_train_epochs=int(self.config['training']['NUM_TRAIN_EPOCHS']),
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=int(self.config['training']['LOGGING_STEPS']),
            save_strategy="epoch",
            learning_rate=float(self.config['training']['LEARNING_RATE']),
            bf16=True,
            lr_scheduler_type="cosine",
            dataset_text_field='',
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            save_only_model=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0, 
            # FSDP configuration for distributed training
            fsdp='full_shard auto_wrap' if self.world_size > 1 else '',
            fsdp_config={
                'fsdp_transformer_layer_cls_to_wrap': ['Gemma3DecoderLayer'],
                'fsdp_activation_checkpointing': False,
                **self.config['fsdp']
            } if self.world_size > 1 else None,
            # Distributed training settings
            local_rank=self.local_rank,
            ddp_find_unused_parameters=False,
        )
    
    def train(self):
        """Execute the training loop"""
        try:
            self.print_rank0("="*50)
            self.print_rank0("Starting Distributed Training")
            self.print_rank0(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            self.print_rank0(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            self.print_rank0(f"Number of GPUs: {self.world_size}")
            self.print_rank0(f"Rank: {self.rank} (Local: {self.local_rank})")
            self.print_rank0("="*50)

            self.print_rank0("\n[STEP 1] Loading model and processor...")
            self.load_model_and_processor()
            
            self.print_rank0("\n[STEP 2] Loading dataset...")
            self.print_rank0(f"Dataset ID: {self.config['dataset']['DATASET_ID']}")
            raw_dataset = load_dataset(
                self.config['dataset']['DATASET_ID'], 
                split="train"
            )
            
            # Limit samples if specified
            num_samples = self.config['dataset'].get('NUM_SAMPLES', len(raw_dataset))
            if num_samples < len(raw_dataset):
                raw_dataset = raw_dataset.select(range(num_samples))
            
            self.print_rank0(f"✓ Raw dataset loaded with {len(raw_dataset)} samples")
            
            dataset = CustomDataset(
                raw_dataset, 
                self.processor, 
                img_size=self.config['model']['IMG_SIZE'], 
                max_length=self.config['model']['MAX_SEQ_LENGTH']
            )
            self.print_rank0(f"✓ Custom dataset created with {len(dataset)} samples")
        
            self.print_rank0("\n[STEP 3] Initializing trainer...")
            training_args = self.get_training_args()
            
            # Initialize trainer
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                peft_config=self._get_peft_config(),
                data_collator=dataset.collate_fn,
            )
            self.print_rank0("✓ Trainer initialized successfully")

            # Ensure all parameters are in bfloat16
            for name, param in trainer.model.named_parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.bfloat16)

            self.print_rank0("\n[STEP 4] Starting training loop...")
            
            # Clear CUDA cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            trainer.train()
            self.print_rank0("✓ Training completed successfully")

            # Synchronize before saving
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.print_rank0("\n[STEP 5] Saving the final adapter...")
            adapter_path = os.path.join(
                self.config['training']['OUTPUT_DIR'], 
                "final_adapter"
            )
            trainer.save_model(adapter_path)
            self.print_rank0(f"✓ Adapter saved to {adapter_path}")

            # Only merge on the main process to avoid redundant work
            if self.is_main_process():
                self.print_rank0("\n[STEP 6] Merging adapter with base model (main process only)...")

                base_model = AutoModelForImageTextToText.from_pretrained(
                    self.config['model']['BASE_MODEL_ID'],
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

                model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)

                # Merge the adapter weights into the base model
                self.print_rank0("Merging LoRA layers...")
                merged_model = model_to_merge.merge_and_unload()
                self.print_rank0("✓ LoRA layers merged successfully")

                # Save the merged model
                merged_model_path = os.path.join(
                    self.config['training']['OUTPUT_DIR'], 
                    "final_merged_model"
                )
                merged_model.save_pretrained(merged_model_path)
                
                # Also save the tokenizer for easy future use
                self.processor.tokenizer.save_pretrained(merged_model_path)

                self.print_rank0(f"✓ Merged model saved to {merged_model_path}")

            # Wait for all processes to complete
            if self.world_size > 1:
                dist.barrier()

        except Exception as e:
            print(f"\n✗ Error in training (rank {self.rank}): {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise

    def run(self):
        """Run training with distributed setup"""
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")
        
        self.print_rank0(f"Starting training on {self.world_size} GPU(s)...")
        self.train()


def main():
    """
    Main entry point for distributed training.
    This can be run directly or launched with torchrun/Vertex AI.
    """
    try:
        # Set environment variables for better CUDA error handling
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA calls for better error traces
        os.environ['TORCH_USE_CUDA_DSA'] = '1'     # Device-side assertions
        
        # Initialize distributed training
        is_distributed = setup_distributed()
        
        # Create and run trainer
        trainer = DistributedTrainer()
        trainer.run()
        
        # Print final message only from main process
        if trainer.is_main_process():
            print("\n" + "="*50)
            print("Training completed successfully!")
            print("="*50)
        
    except Exception as e:
        rank = int(os.environ.get('RANK', '0'))
        print(f"Training failed on rank {rank}: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        # Clean up distributed resources
        cleanup_distributed()


if __name__ == "__main__":
    main()
