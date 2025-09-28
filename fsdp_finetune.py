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


class DtypeCorrectingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        peft_config = kwargs.get("peft_config")
        super().__init__(*args, **kwargs)
        # After PEFT model is created, but before FSDP wrapping, cast dtypes
        if peft_config and self.args.bf16:
            self.model.to(torch.bfloat16)

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
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print(f"Detected {available_gpus} available GPU(s)")
        
        # For single GPU or CPU, disable FSDP to avoid distributed setup issues
        if available_gpus <= 1 and self.use_fsdp:
            print("Single GPU/CPU detected. Disabling FSDP for simpler training.")
            self.use_fsdp = False
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            return self.local_rank, self.world_size, self.rank
        
        if self.use_fsdp:
            has_torchrun_env = (
                os.environ.get("LOCAL_RANK") is not None
                and os.environ.get("RANK") is not None
                and os.environ.get("WORLD_SIZE") is not None
            )

            if not has_torchrun_env:
                print(
                    "Multiple GPUs detected but distributed launch arguments were not provided. "
                    "Please launch with `torchrun --nproc_per_node=<num_gpus>` to enable FSDP. "
                    "Falling back to single-process training without FSDP."
                )
                self.use_fsdp = False
                self.local_rank = 0
                self.world_size = 1
                self.rank = 0
                return self.local_rank, self.world_size, self.rank

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

        print(f"Using world_size: {self.world_size}, rank: {self.rank}, local_rank: {self.local_rank}")

        # Only initialize if using FSDP
        if self.use_fsdp and not dist.is_initialized():
            print("Initializing distributed process group...")
            try:
                backend = "nccl"
                nccl_available = hasattr(dist, "is_nccl_available") and dist.is_nccl_available()
                if os.name == "nt" or not nccl_available:
                    backend = "gloo"

                dist.init_process_group(
                    backend=backend,
                    rank=self.rank,
                    world_size=self.world_size
                )
                print(f"Successfully initialized distributed training with {self.world_size} GPU(s)")
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")
                print("Falling back to single GPU training without FSDP...")
                self.use_fsdp = False
                self.local_rank = 0
                self.world_size = 1
                self.rank = 0
        
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
        print("Loading model and processor...")
        
        # IMPORTANT: FSDP cannot shard bitsandbytes 4-bit quantized base weights.
        # When FSDP is enabled, we load the model on CPU without quantization and let FSDP shard it.
        # When FSDP is disabled (QLoRA path), we use 4-bit quantization and leverage HF device_map for model parallelism.

        using_fsdp_path = bool(self.use_fsdp)

        if using_fsdp_path:
            print("FSDP is enabled: loading UN-quantized model on CPU to allow sharding across GPUs")
            device_map = 'cpu'
            quantization_config = None
        else:
            print("FSDP is disabled: using QLoRA with 4-bit quantization and device_map for model parallelism")
            # With multiple GPUs, use 'auto' to spread layers across GPUs and reduce per-GPU memory
            device_map = "auto" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else ({'': local_rank} if torch.cuda.is_available() else 'cpu')
            quantization_config = self._get_quantization_config()
        
        print(f"Loading model: {self.config['model']['BASE_MODEL_ID']}")
        model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['BASE_MODEL_ID'],
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully")

        model.config.use_cache = False

        print(f"Loading processor: {self.config['model']['CHAT_MODEL_ID']}")
        # Load processor
        processor = AutoProcessor.from_pretrained(
            self.config['model']['CHAT_MODEL_ID'],
            trust_remote_code=True
        )
        print("Processor loaded successfully")

        print("Applying PEFT configuration...")
        if using_fsdp_path:
            # Standard LoRA on full-precision weights; we attach LoRA before FSDP wrapping
            model = get_peft_model(model, self._get_peft_config())
            # Ensure uniform dtype for all params before FSDP flattening
            model.to(torch.bfloat16)
            print("LoRA attached (non-quantized) and model cast to bfloat16")
        else:
            # QLoRA path: prepare for k-bit training; LoRA layers will be attached by the Trainer via peft_config
            model = prepare_model_for_kbit_training(model)
            print("Model prepared for QLoRA (4-bit)")

        # # Freeze all parameters except LoRA layers
        # trainable_params = 0
        # total_params = 0
        # print("Configuring trainable parameters...")
        # with torch.no_grad():
        #     for name, param in model.named_parameters():
        #         total_params += param.numel()
        #         if ".lora_A." in name or ".lora_B." in name:
        #             param.requires_grad_(True)
        #             trainable_params += param.numel()
        #         else:
        #             param.requires_grad_(False)
        
        # print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()
        print("Model configuration completed")

        self.model = model
        self.processor = processor
    
    def _setup_fsdp_policies(self):
        """Set up FSDP policies and configurations"""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up FSDP policies")
            
        self.auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Gemma3DecoderLayer},
        )

        self.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
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
        if self.model is None:
            raise ValueError("Model must be loaded before applying FSDP")
            
        print("Wrapping model with FSDP...")
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
        print("FSDP wrapping completed")

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
        effective_batch_size = int(self.config['training']['BATCH_SIZE']) // self.world_size
        gradient_accumulation_steps = int(self.config['training']['GRADIENT_ACCUMULATION_STEPS'])

        # Disable HF Trainer-managed FSDP because we handle wrapping manually when self.use_fsdp is True.
        # Also disable in QLoRA path because 4-bit bnb weights cannot be FSDP sharded.
        return SFTConfig(
            output_dir=self.config['training']['OUTPUT_DIR'],
            num_train_epochs=int(self.config['training']['NUM_TRAIN_EPOCHS']),
            per_device_train_batch_size=effective_batch_size,
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
            fsdp="",
            fsdp_config=None,
        )
    
    def collate_fn(self, batch):
        if self.processor is None:
            raise ValueError("Processor not initialized")

        # Pre-extract all valid samples to avoid multiple iterations
        valid_samples = []
        for example in batch:
            user_message = next(
                (msg for msg in example.get("messages", []) if msg.get("role") == "user"), None
            )
            
            if not user_message:
                continue
                
            image_content = next(
                (content for content in user_message.get("content", []) if content.get("type") == "image"), None
            )
            
            if not image_content or not image_content.get("image"):
                if self.rank == 0:
                    print("Warning: Skipping a sample due to a missing image.")
                continue
                
            chat_text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            
            valid_samples.append((chat_text.strip(), image_content["image"]))

        if not valid_samples:
            return {}

        # Batch process all valid samples at once
        texts, images = zip(*valid_samples)
        
        try:
            inputs = self.processor(
                text=list(texts), 
                images=list(images), 
                return_tensors="pt", 
                padding=True, 
                max_length=512, 
                truncation=True
            )
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Batch processing failed: {e}")
            return {}

        # Create labels efficiently using in-place operations where possible
        labels = inputs["input_ids"].clone()
        pad_token_mask = (labels == self.processor.tokenizer.pad_token_id)
        attention_mask_zero = (inputs['attention_mask'] == 0)
        
        labels[pad_token_mask | attention_mask_zero] = -100
        inputs['labels'] = labels
        
        return inputs
    
    # def collate_fn(self, batch):
    #     if self.processor is None:
    #         raise ValueError("Processor not initialized")

    #     # Process each sample individually and collect the processed outputs
    #     processed_batches = []
    #     for example in batch:
    #         # Extract text and image from the single example
    #         chat_text = self.processor.apply_chat_template(
    #             example["messages"], add_generation_prompt=False, tokenize=False
    #         )
            
    #         user_message = next(
    #             (msg for msg in example.get("messages", []) if msg.get("role") == "user"), None
    #         )
            
    #         image_value = None
    #         if user_message:
    #             image_content = next(
    #                 (content for content in user_message.get("content", []) if content.get("type") == "image"), None
    #             )
    #             if image_content:
    #                 image_value = image_content.get("image")

    #         if image_value is None:
    #             if self.rank == 0:
    #                 print("Warning: Skipping a sample due to a missing image.")
    #             continue

    #         # The processor expects lists, so wrap text and image in lists
    #         try:
    #             inputs = self.processor(
    #                 text=[chat_text.strip()], 
    #                 images=[image_value], 
    #                 return_tensors="pt", 
    #                 padding=True, 
    #                 max_length=512, 
    #                 truncation=True
    #             )
    #             processed_batches.append(inputs)
    #         except Exception as e:
    #             if self.rank == 0:
    #                 print(f"Warning: Skipping a sample due to processor error: {e}")
    #             continue

    #     if not processed_batches:
    #         # Return an empty dictionary if all samples in the batch were skipped
    #         return {}

    #     # Manually collate the processed batches
    #     # This part needs to handle padding between samples of different lengths
    #     # A simple approach is to find the max length and pad everything to it.
    #     max_len = max(b["input_ids"].shape[1] for b in processed_batches)

    #     input_ids_list = []
    #     attention_mask_list = []
    #     pixel_values_list = []

    #     for b in processed_batches:
    #         # Pad input_ids and attention_mask to the max length in the batch
    #         pad_len = max_len - b["input_ids"].shape[1]
    #         if pad_len > 0:
    #             # Use the processor's pad_token_id for padding
    #             pad_tensor_ids = torch.full((1, pad_len), self.processor.tokenizer.pad_token_id, dtype=torch.long)
    #             input_ids = torch.cat([b["input_ids"], pad_tensor_ids], dim=1)
                
    #             pad_tensor_mask = torch.full((1, pad_len), 0, dtype=torch.long)
    #             attention_mask = torch.cat([b["attention_mask"], pad_tensor_mask], dim=1)
    #         else:
    #             input_ids = b["input_ids"]
    #             attention_mask = b["attention_mask"]

    #         input_ids_list.append(input_ids)
    #         attention_mask_list.append(attention_mask)
    #         pixel_values_list.append(b["pixel_values"])

    #     # Stack the tensors to create the final batch
    #     final_batch = {
    #         "input_ids": torch.cat(input_ids_list, dim=0),
    #         "attention_mask": torch.cat(attention_mask_list, dim=0),
    #         "pixel_values": torch.cat(pixel_values_list, dim=0),
    #     }

    #     # Create labels
    #     labels = final_batch["input_ids"].clone()
    #     labels[labels == self.processor.tokenizer.pad_token_id] = -100
    #     labels[final_batch['attention_mask'] == 0] = -100
    #     final_batch['labels'] = labels
        
    #     return final_batch

    def train(self): 
        try:
            print("="*50)
            print("Starting training")
            print(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            print(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            print(f"Using FSDP: {self.use_fsdp}")
            print(f"Batch size: {self.config['training']['BATCH_SIZE']}")
            print(f"Epochs: {self.config['training']['NUM_TRAIN_EPOCHS']}")
            print("="*50)

            print("\n[STEP 1] Initializing distributed training...")
            local_rank, world_size, rank = self._init_distributed()
            print(f"✓ Distributed setup complete: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            
            print("\n[STEP 2] Setting up compute device...")
            # Ensure we're using the correct GPU
            if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                print(f"✓ Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
            else:
                device = torch.device("cpu")
                print("✓ Using CPU")

            print("\n[STEP 3] Loading model and processor...")
            self.load_model_and_processor(local_rank)
            print("✓ Model and processor loaded successfully")
            
            # Only apply FSDP if enabled
            if self.use_fsdp:
                print("\n[STEP 4] Setting up FSDP...")
                self._setup_fsdp_policies()
                self._apply_fsdp()
                print("✓ FSDP setup completed")
            else:
                print("\n[STEP 4] Skipping FSDP (disabled in config)")

            print("\n[STEP 5] Loading dataset...")
            print(f"Dataset ID: {self.config['dataset']['DATASET_ID']}")
            raw_dataset = load_dataset(self.config['dataset']['DATASET_ID'], split="train")
            try:
                dataset_size = len(raw_dataset)  # type: ignore
                print(f"✓ Raw dataset loaded with {dataset_size} samples")
            except:
                print("✓ Raw dataset loaded (size unknown for streaming dataset)")
            
            dataset = CustomDataset(raw_dataset)
            print(f"✓ Custom dataset created with {len(dataset)} samples")

            print("\n[STEP 6] Initializing trainer...")
            training_args = self.get_training_args()
            print(f"Training args: output_dir={training_args.output_dir}, epochs={training_args.num_train_epochs}")
            
            # Initialize trainer
            # trainer = SFTTrainer(
            #     model=self.model,
            #     args=training_args,
            #     train_dataset=dataset,
            #     peft_config=self._get_peft_config(),
            #     # processing_class=self.processor,
            #     data_collator=self.collate_fn,
            # )

            # Initialize the custom trainer
            if self.use_fsdp:
                # LoRA already attached; do NOT pass peft_config to avoid double application
                trainer = DtypeCorrectingSFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=self.collate_fn,
                )
            else:
                # QLoRA path: pass peft_config so trainer attaches LoRA on quantized base
                trainer = DtypeCorrectingSFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=dataset,
                    peft_config=self._get_peft_config(),
                    data_collator=self.collate_fn,
                )
            print("✓ Trainer initialized successfully")

            print("\n[STEP 7] Starting training loop...")
            trainer.train()
            print("✓ Training completed successfully")

            # Save model and merge (only on rank 0 for multi-GPU)
            if self.rank == 0:
                print("\n[STEP 8] Saving and merging model...")
                trainer.save_model()
                print("✓ Model saved successfully")

                try:
                    print("Merging model...")
                    merged_model = get_merged_model(
                        self.config['model']['BASE_MODEL_ID'],
                        self.config['training']['OUTPUT_DIR'],
                        self.config['training']['MERGED_MODEL_DIR']
                    )
                    print(f"✓ Model saved to: {self.config['training']['OUTPUT_DIR']}")
                    print(f"✓ Merged model saved to: {self.config['training']['MERGED_MODEL_DIR']}")
                except Exception as e:
                    print(f"Warning: Model merging failed: {e}")
                    print("Training was successful, but merged model not created")

        except Exception as e:
            print(f"\n✗ Error in training: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
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