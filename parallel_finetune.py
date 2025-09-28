import os
from pathlib import Path
import traceback
import sys


from data_preprocessing import CustomDataset
from inference import get_merged_model

import torch
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import yaml
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RMSNorm
from typing import Callable

class CustomSFTTrainer(SFTTrainer):
    def _fsdp_qlora_plugin_updates(self):
        def custom_policy(module, recurse, nonwrapped_numel) -> bool:
            return isinstance(module, Gemma3DecoderLayer)
        
        self.accelerator.state.fsdp_plugin.auto_wrap_policy = custom_policy


class Trainer:
    """Handles FSDP training setup and execution"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.config = {}

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
        
        env_vars = self.config.get('environment', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)

    def _get_quantization_config(self):
        """Get quantization configuration for 4-bit training"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_quant_storage=torch.bfloat16,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_storage=torch.float32,
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
    
    def _cast_mixed_precision_to_bf16(self, model):
        """Ensure all parameters have consistent bf16 dtype for FSDP compatibility"""
        print("Converting all model parameters to bfloat16...")
        
        # Convert all parameters to bfloat16
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        
        # Convert all buffers to bfloat16
        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.float32:
                buffer.data = buffer.data.to(torch.bfloat16)
        
        # Ensure specific modules are in bfloat16
        for module in model.modules():
            if isinstance(module, (torch.nn.LayerNorm, Gemma3RMSNorm, torch.nn.Embedding)):
                module.to(torch.bfloat16)
        
        print("✓ All parameters converted to bfloat16")

    def load_model_and_processor(self):
        print(f"Loading model: {self.config['model']['BASE_MODEL_ID']}")
        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['BASE_MODEL_ID'],
            quantization_config=self._get_quantization_config(),
            # dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("Model loaded successfully")

        model.config.use_cache = False

        print(f"Loading processor: {self.config['model']['CHAT_MODEL_ID']}")
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config['model']['CHAT_MODEL_ID'],
            trust_remote_code=True
        )
        print("Processor loaded successfully")

        print("Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(model)

        # Cast all parameters to bfloat16 BEFORE trainer initialization
        # self._cast_mixed_precision_to_bf16(self.model)

        print("Model configuration completed")

    def get_training_args(self):
        """Get training arguments configuration"""
        effective_batch_size = int(self.config['training']['BATCH_SIZE']) // int(os.environ["WORLD_SIZE"])
        gradient_accumulation_steps = int(self.config['training']['GRADIENT_ACCUMULATION_STEPS'])

        return SFTConfig(
            output_dir=self.config['training']['OUTPUT_DIR'],
            num_train_epochs=int(self.config['training']['NUM_TRAIN_EPOCHS']),
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=int(self.config['training']['LOGGING_STEPS']),
            save_strategy="epoch",
            learning_rate=float(self.config['training']['LEARNING_RATE']),
            # bf16=True,
            lr_scheduler_type="cosine",
            dataset_text_field='',
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            save_only_model=True,
            dataloader_pin_memory=False,
            # Tell the trainer to use FSDP
            fsdp='full_shard',
            fsdp_config={
                'fsdp_transformer_layer_cls_to_wrap': [Gemma3DecoderLayer],
                **self.config['fsdp']
            }
        )
    
    def train(self): 
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        try:
            print("="*50)
            print("Starting training")
            print(f"Model: {self.config['model']['BASE_MODEL_ID']}")
            print(f"Dataset: {self.config['dataset']['DATASET_ID']}")
            print(f"Batch size: {self.config['training']['BATCH_SIZE']}")
            print(f"Epochs: {self.config['training']['NUM_TRAIN_EPOCHS']}")
            print("="*50)

            print("\n[STEP 1] Loading model and processor...")
            self.load_model_and_processor()
            print("✓ Model and processor loaded successfully")

            print("\n[STEP 2] Loading dataset...")
            print(f"Dataset ID: {self.config['dataset']['DATASET_ID']}")
            raw_dataset = load_dataset(self.config['dataset']['DATASET_ID'], split="train")
            try:
                dataset_size = len(raw_dataset)  # type: ignore
                print(f"✓ Raw dataset loaded with {dataset_size} samples")
            except:
                print("✓ Raw dataset loaded (size unknown for streaming dataset)")
            
            dataset = CustomDataset(raw_dataset, self.processor)
            print(f"✓ Custom dataset created with {len(dataset)} samples")
        
            print("\n[STEP 3] Initializing trainer...")
            training_args = self.get_training_args()
            print(f"Training args: output_dir={training_args.output_dir}, epochs={training_args.num_train_epochs}")
            
            # Initialize trainer
            trainer = CustomSFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                peft_config=self._get_peft_config(),
                data_collator=dataset.collate_fn,
            )
            print("✓ Trainer initialized successfully")

            # # Ensure LoRA parameters are in correct dtype
            # print("Ensuring LoRA parameters are in bfloat16...")
            # for name, module in trainer.model.named_modules():
            #     if "lora_" in name:
            #         module.to(torch.bfloat16)
            # print("✓ LoRA parameters set to bfloat16")

            print("\n[STEP 4] Starting training loop...")
            trainer.train()
            print("✓ Training completed successfully")

            print("\n[STEP 5] Saving the final adapter...")
            adapter_path = os.path.join(self.config['training']['OUTPUT_DIR'], "final_adapter")
            trainer.save_model(adapter_path)
            print(f"✓ Adapter saved to {adapter_path}")

            if trainer.is_world_process_zero():
                print("\n[STEP 6] Merging adapter with base model on main process (rank 0)...")

                model_to_merge = PeftModel.from_pretrained(self.model, adapter_path)

                # Merge the adapter weights into the base model
                print("Merging LoRA layers...")
                merged_model = model_to_merge.merge_and_unload()
                print("✓ LoRA layers merged successfully")

                # Save the merged model
                merged_model_path = os.path.join(self.config['training']['OUTPUT_DIR'], "final_merged_model")
                merged_model.save_pretrained(merged_model_path)
                
                # Also save the tokenizer for easy future use
                self.processor.tokenizer.save_pretrained(merged_model_path)

                print(f"✓ Merged model saved to {merged_model_path}")


        except Exception as e:
            print(f"\n✗ Error in training: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise

    
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