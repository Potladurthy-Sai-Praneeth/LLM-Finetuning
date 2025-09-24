from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText , AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import os
import torch
import argparse
from PIL import Image
import requests
from data_preprocessing import *



def get_merged_model(model_id,output_dir, merged_model_dir):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_model = PeftModel.from_pretrained(model, output_dir, torch_dtype=torch.bfloat16)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")
    return merged_model

def get_inference(sample, model, processor):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": sample['prompt'].format(product=sample["product_name"], category=sample["category"])},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=sample["image"],
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)
    # Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=512, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and save a PEFT model.")
    parser.add_argument("--output_folder", type=str, required=False, help="Path to the folder containing the PEFT model.")
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--prompt', type=str, help='Input prompt to the model')

    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.output_folder, device_map="auto", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.output_folder)

    image = Image.open(args.image_path).convert("RGB")
    
    sample = {
        "image": image,
        "question": args.prompt,
    }

    output = get_inference(sample, model, processor)
    print("Output:", output)


