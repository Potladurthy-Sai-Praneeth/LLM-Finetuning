import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self,dataset,processor,img_size=448,max_length=256) -> None:
        super().__init__()

        self.dataset = dataset
        self.processor = processor
        self.system_message = "You are an expert in Medical Diagnosis and treatment. You are given a image. You need to provide a description of the disease or anamoly or sometimes suggest possible treatments analysing the image."
        self.system_message_content = [{"type": "text", "text": self.system_message}]

        self.index_mapping = []
        self.img_size = int(img_size), int(img_size)
        self.max_length = int(max_length)

        for sample_idx, point in enumerate(self.dataset):
            questions = point['questions']
            for qa_idx in range(len(questions)):
                self.index_mapping.append((sample_idx, qa_idx))
        
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        sample_idx, qa_idx = self.index_mapping[idx]
        point = self.dataset[sample_idx]
        question = point['questions'][qa_idx]
        answer = point['answers'][qa_idx]
        image = point['image']
        
        # Ensure image is PIL Image and in RGB mode
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        
        image = image.resize(self.img_size)
        
        return self.format_data(image,question,answer)
    

    #Gemma 3 
    def format_data(self,image,question,answer):
        '''
        This function returns a dictionary formatted for training/fine-tuning a multimodal model.
        '''
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_message_content,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ],
        }

 
    

    def collate_fn(self, batch):
        if self.processor is None:
            raise ValueError("Processor not initialized")

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
                print("Warning: Skipping a sample due to a missing image.")
                continue
            
            valid_samples.append((example["messages"], image_content["image"]))

        if not valid_samples:
            # Return a properly formatted empty batch instead of empty dict
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)
            }

        # Process samples individually to avoid batch size mismatch
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        
        for messages, image in valid_samples:
            try:
                # Apply chat template to get formatted text
                chat_text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )
                
                # Process single sample
                inputs = self.processor(
                    text=chat_text.strip(), 
                    images=image, 
                    return_tensors="pt", 
                    padding=False,  # We'll pad the batch later
                    max_length=self.max_length, 
                    truncation=True
                )
                
                all_input_ids.append(inputs["input_ids"].squeeze(0))
                all_attention_masks.append(inputs["attention_mask"].squeeze(0))
                
                # Capture pixel_values if present (for vision models)
                if "pixel_values" in inputs:
                    all_pixel_values.append(inputs["pixel_values"].squeeze(0))
                
            except Exception as e:
                print(f"Warning: Sample processing failed: {e}")
                continue
        
        if not all_input_ids:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)
            }
        
        # Pad sequences to the same length
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence(all_attention_masks, batch_first=True, padding_value=0)
        
        # Stack pixel values if available
        pixel_values = torch.stack(all_pixel_values) if all_pixel_values else None

        labels = input_ids.clone()
        
        # Get image token ID once and reuse
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.tokenizer.special_tokens_map["boi_token"]
        )

        # Create combined mask for tokens to ignore in loss computation
        mask = (
            (labels == self.processor.tokenizer.pad_token_id) |
            (labels == image_token_id) |
            (labels == 262144)
        )
        labels[mask] = -100
        del mask
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add pixel_values if available
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        return result
    
# =========================================================================
#                             For Pali-Gemma
# =========================================================================
    # def format_data(self, image, question, answer):
    #     # PaliGemma format: <image> question\nanswer
    #     return {
    #         "image": image,
    #         "text": f"<image> {question}\n{answer}"
    #     }

    # def collate_fn(self, batch):
    #     if self.processor is None:
    #         raise ValueError("Processor not initialized")

    #     images = []
    #     texts = []

    #     for example in batch:
    #         img = example.get("image", None)
    #         if img is None:
    #             print("Warning: Skipping sample with missing image")
    #             continue

    #         # Image is already processed in __getitem__ as RGB PIL Image
    #         if not isinstance(img, Image.Image):
    #             print(f"Warning: Unexpected image type {type(img)}, skipping")
    #             continue

    #         images.append(img)
            
    #         text = example["text"]
    #         if "<image>" not in text:
    #             text = "<image> " + text
    #         texts.append(text)

    #     if not images:
    #         return {
    #             "input_ids": torch.tensor([], dtype=torch.long),
    #             "attention_mask": torch.tensor([], dtype=torch.long),
    #             "labels": torch.tensor([], dtype=torch.long)
    #         }

    #     inputs = self.processor(
    #         images=images,
    #         text=texts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.max_length
    #     )

    #     labels = inputs["input_ids"].clone()
    #     labels[labels == self.processor.tokenizer.pad_token_id] = -100
    #     inputs["labels"] = labels

    #     return inputs


# def process_dataset(dataset):
#     '''
#     This function processes the dataset to create a list of formatted samples.
#     Each sample contains an image, a question, and an answer.
#     Note: dataset.map is not used as it serializes the image object which is not desired.
#     '''
#     processed_data = []

#     for point in dataset:
#         image,questions,answers = point['image'],point['questions'],point['answers']
#         for pair in zip(questions,answers):
#             question,answer = pair
#             sample = {
#                 "image": image,
#                 "question": question,
#                 "answer": answer
#             }
#             formatted_sample = format_data(sample)
#             processed_data.append(formatted_sample)
    
#     return processed_data
