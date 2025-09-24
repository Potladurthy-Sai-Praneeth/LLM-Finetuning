from datasets import load_dataset
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText , AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig
from data_preprocessing import *
from inference import get_merged_model


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)


load_dotenv()
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-4))
DATASET_ID = os.getenv("DATASET_ID")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "gemma-medical")
MERGED_MODEL_DIR = os.getenv("MERGED_MODEL_DIR", OUTPUT_DIR + "-merged")

    
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)


peft_config = LoraConfig(
    lora_alpha=32, 
    lora_dropout=0.1, 
    r=64, 
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=[  
        "lm_head",
        "embed_tokens",
    ],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


args = SFTConfig(
    output_dir=OUTPUT_DIR,     
    num_train_epochs=NUM_EPOCHS,                         
    per_device_train_batch_size=BATCH_SIZE,              
    gradient_accumulation_steps=5,              
    gradient_checkpointing=True,            
    optim="adamw_torch_fused",                  
    logging_steps=10,                            
    save_strategy="epoch",                       
    learning_rate=LEARNING_RATE,                         
    bf16=True,                                  
    lr_scheduler_type="cosine",                 
    dataset_text_field='',
    dataset_kwargs={"skip_prepare_dataset": True},  
    remove_unused_columns=False,               
)


dataset = load_dataset(DATASET_ID, split="train")

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.train()

trainer.save_model()

merged_model = get_merged_model(BASE_MODEL_ID, OUTPUT_DIR, MERGED_MODEL_DIR)