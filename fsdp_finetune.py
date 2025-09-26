import os
from pathlib import Path

from data_preprocessing import CustomDataset
from inference import get_merged_model

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from functools import partial
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
import yaml


class Trainer:
    """Handles FSDP training setup and execution"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.config = {}
        self.auto_wrap_policy = None
        self.mixed_precision_policy = None
        self.cpu_offload = None
        self.ignore_modules = []
        self.local_rank = 0
        self.world_size = 1
        self.rank = 0
        self.use_fsdp = True  # Default value

        self._load_config()
        self._setup_environment()
        
        # Override FSDP setting from config if available
        self.use_fsdp = self.config.get('training', {}).get('USE_FSDP', True)

    def _load_config(self):
        """Load configuration from YAML file"""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
    
    def _setup_environment(self):
        """Set up environment variables for optimal performance"""
        env_vars = self.config.get('environment', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
    
    def _init_distributed(self):
        """Initialize process group for distributed training."""
        # Auto-detect world size from available GPUs
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(available_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", available_gpus))
        self.rank = int(os.environ.get("RANK", 0))

        print(f"Detected {available_gpus} available GPU(s)")
        print(f"Using world_size: {self.world_size}, rank: {self.rank}, local_rank: {self.local_rank}")

        # Only initialize if using FSDP or multi-GPU
        if self.use_fsdp and not dist.is_initialized():
            # Only initialize distributed if we have multiple GPUs or explicitly want FSDP
            if self.world_size > 1:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=self.rank,
                    world_size=self.world_size
                )
                print(f"Initialized distributed training with {self.world_size} GPUs")
            else:
                print("Single GPU detected, but FSDP enabled. Initializing process group for FSDP...")
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=0,
                    world_size=1
                )
        
        return self.local_rank, self.world_size, self.rank

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

    def load_model_and_processor(self, local_rank):
        """Load and configure model and processor for training"""
        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['BASE_MODEL_ID'],
            quantization_config=self._get_quantization_config(),
            dtype=torch.bfloat16,
            device_map={'': local_rank},
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
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self._get_peft_config())

        # Freeze all parameters except LoRA layers
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_A." in name or ".lora_B." in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        self.model = model
        self.processor = processor
    
    def _setup_fsdp_policies(self):
        """Set up FSDP policies and configurations"""
        self.auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Gemma3DecoderLayer},
        )

        self.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            cast_forward_inputs=True,
        )

        self.cpu_offload = CPUOffload(offload_params=True)

        # Build ignored modules list for FSDP
        self.ignore_modules = []
        for module in self.model.modules():
            params = list(module.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                self.ignore_modules.append(module)
    
    def _apply_fsdp(self):
        """Apply FSDP wrapping to the model"""
        self.model = FSDP(
            self.model,
            auto_wrap_policy=self.auto_wrap_policy,
            mixed_precision=self.mixed_precision_policy,
            device_id=self.local_rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            limit_all_gathers=True,
            use_orig_params=True,
            cpu_offload=self.cpu_offload,
            ignored_modules=self.ignore_modules, 
        )

        # Apply activation checkpointing AFTER FSDP wrapping
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, Gemma3DecoderLayer)
        apply_activation_checkpointing(
            self.model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
    
    def get_training_args(self):
        """Get training arguments configuration"""
        effective_batch_size = self.config['training']['BATCH_SIZE'] // self.world_size
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

    def train(self): 
        try:
            print("Starting training")
            print(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            print(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            print(f"Using FSDP: {self.use_fsdp}")

            local_rank, world_size, rank = self._init_distributed()
            
            # Ensure we're using the correct GPU
            if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                print(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
            else:
                device = torch.device("cpu")
                print("Using CPU")

            self.load_model_and_processor(local_rank)
            
            # Only apply FSDP if enabled
            if self.use_fsdp:
                self._setup_fsdp_policies()
                self._apply_fsdp()

            dataset = CustomDataset(load_dataset(self.config['dataset']['DATASET_ID'], split="train"))

            # Initialize trainer
            trainer = SFTTrainer(
                model=self.model,
                args=self.get_training_args(),
                train_dataset=dataset,
                peft_config=self._get_peft_config(),
                processing_class=self.processor,
                data_collator=self.collate_fn,
            )

            trainer.train()

            # Save model and merge (only on rank 0 for multi-GPU)
            if self.rank == 0:
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
        finally:
            # Clean up distributed process group
            if self.use_fsdp and dist.is_initialized():
                dist.destroy_process_group()

    def run(self):
        """Run training"""
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")
        
        print("Starting FSDP training...")
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