import os
from pathlib import Path
import traceback
import sys
import json

from data_preprocessing import CustomDataset
from inference import get_merged_model

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
import deepspeed


class Trainer:
    """Handles DeepSpeed training setup and execution"""

    def __init__(self):
        self.processor = None
        self.model = None
        self.config = {}
        
        # Initialize DeepSpeed distributed training early for QLoRA
        if not dist.is_initialized():
            deepspeed.init_distributed()
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        self._load_config()
            

    def _load_config(self):
        """Load configuration from YAML file"""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Load DeepSpeed configuration
        ds_config_path = script_dir / "ds_config.json"
        if not ds_config_path.exists():
            raise FileNotFoundError(f"DeepSpeed config file not found at {ds_config_path}")

        with open(ds_config_path, "r") as file:
            self.ds_config = json.load(file)

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
        """Load processor and create model for DeepSpeed with proper device handling"""
        print(f"Loading processor: {self.config['model']['CHAT_MODEL_ID']}")
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config['model']['CHAT_MODEL_ID'],
            trust_remote_code=True
        )
        print("Processor loaded successfully")
        
        print(f"Loading model on CPU: {self.config['model']['BASE_MODEL_ID']}")
        
        # Load model on CPU first to avoid GPU memory issues  
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}
        model = AutoModelForImageTextToText.from_pretrained(
                self.config['model']['BASE_MODEL_ID'],
                quantization_config=self._get_quantization_config(),
                dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
        
        model.config.use_cache = False
        print("Model loaded on CPU successfully")
        
        # Prepare model for k-bit training
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        print("Model preparation completed")
        return model


    def get_training_args(self):
        """Get training arguments configuration for DeepSpeed"""
        per_device_batch_size = int(self.config['training']['BATCH_SIZE'])
        gradient_accumulation_steps = int(self.config['training']['GRADIENT_ACCUMULATION_STEPS'])

        return SFTConfig(
            output_dir=self.config['training']['OUTPUT_DIR'],
            num_train_epochs=int(self.config['training']['NUM_TRAIN_EPOCHS']),
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(self.config['training']['LOGGING_STEPS']),
            save_strategy="epoch",
            bf16=True,
            dataset_text_field='',
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            save_only_model=True,
            dataloader_pin_memory=False,
            deepspeed=self.ds_config,
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
            ddp_find_unused_parameters=False,
            report_to=None,  
        )

    def train(self):
        try:
            print("="*50)
            print("Starting DeepSpeed training")
            print(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            print(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            print(f"Batch size: {self.config['training']['BATCH_SIZE']}")
            print(f"Epochs: {self.config['training']['NUM_TRAIN_EPOCHS']}")
            print("="*50)

            print("\n[STEP 1] Loading model and processor...")
            model = self.load_model_and_processor()
            print("✓ Model and processor loaded successfully")

            print("\n[STEP 2] Loading dataset...")
            print(f"Dataset ID: {self.config['dataset']['DATASET_ID']}")
            raw_dataset = load_dataset(self.config['dataset']['DATASET_ID'], split="train")
            print("✓ Raw dataset loaded successfully")

            dataset = CustomDataset(raw_dataset, self.processor)
            print(f"✓ Custom dataset created with {len(dataset)} samples")

            print("\n[STEP 3] Initializing trainer with DeepSpeed...")
            training_args = self.get_training_args()
            print(f"Training args: output_dir={training_args.output_dir}, epochs={training_args.num_train_epochs}")

            # Initialize trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=self._get_peft_config(),
                data_collator=dataset.collate_fn,
            )
            print("✓ Trainer initialized successfully")

            print("\n[STEP 4] Starting training loop...")
            trainer.train()
            print("✓ Training completed successfully")

            print("\n[STEP 5] Saving the final adapter...")
            adapter_path = os.path.join(self.config['training']['OUTPUT_DIR'], "final_adapter")
            trainer.save_model(adapter_path)
            print(f"✓ Adapter saved to {adapter_path}")

            # Check if this is the main process (rank 0) for model merging
            if trainer.is_world_process_zero():
                print("\n[STEP 6] Merging adapter with base model on main process (rank 0)...")

                # For DeepSpeed, we need to gather the model from all processes first
                trainer.model.save_pretrained(adapter_path)

                base_model = AutoModelForImageTextToText.from_pretrained(
                    self.config['model']['BASE_MODEL_ID'],
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

                model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)

                # Merge the adapter weights into the base model
                print("Merging LoRA layers...")
                merged_model = model_to_merge.merge_and_unload()
                print("✓ LoRA layers merged successfully")

                # Save the merged model
                merged_model_path = os.path.join(self.config['training']['OUTPUT_DIR'], "final_merged_model")
                merged_model.save_pretrained(merged_model_path)

                # Also save the tokenizer for easy future use
                if self.processor and hasattr(self.processor, 'tokenizer'):
                    self.processor.tokenizer.save_pretrained(merged_model_path)

                print(f"✓ Merged model saved to {merged_model_path}")

        except Exception as e:
            print(f"\n✗ Error in training: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise

    def run(self):
        """Run training with DeepSpeed and QLoRA"""
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")

        print(f"QLoRA + DeepSpeed training initialized")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        if dist.is_initialized():
            print(f"Distributed training - World size: {dist.get_world_size()}, Local rank: {dist.get_rank()}")
        print("Starting DeepSpeed training with QLoRA...")
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
