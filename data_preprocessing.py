import os
from torch.utils.data import Dataset
from PIL import Image
import torch


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
        image = image.resize(self.img_size)
        
        return self.format_data(image,question,answer)
    

    #Gemma 3 
    # def format_data(self,image,question,answer):
    #     '''
    #     This function returns a dictionary formatted for training/fine-tuning a multimodal model.
    #     '''
    #     return {
    #         "messages": [
    #             {
    #                 "role": "system",
    #                 "content": self.system_message_content,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": question},
    #                     {"type": "image", "image": image},
    #                 ],
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": [{"type": "text", "text": answer}],
    #             },
    #         ],
    #     }

 
    

    # def collate_fn(self, batch):
    #     if self.processor is None:
    #         raise ValueError("Processor not initialized")

    #     valid_samples = []
    #     for example in batch:
    #         user_message = next(
    #             (msg for msg in example.get("messages", []) if msg.get("role") == "user"), None
    #         )
            
    #         if not user_message:
    #             continue
                
    #         image_content = next(
    #             (content for content in user_message.get("content", []) if content.get("type") == "image"), None
    #         )
            
    #         if not image_content or not image_content.get("image"):
    #             # if self.rank == 0:
    #             print("Warning: Skipping a sample due to a missing image.")
    #             continue
                
    #         chat_text = self.processor.apply_chat_template(
    #             example["messages"], add_generation_prompt=False, tokenize=False
    #         )
            
    #         valid_samples.append((chat_text.strip(), image_content["image"]))

    #     if not valid_samples:
    #         # Return a properly formatted empty batch instead of empty dict
    #         return {
    #             "input_ids": torch.tensor([], dtype=torch.long),
    #             "attention_mask": torch.tensor([], dtype=torch.long),
    #             "labels": torch.tensor([], dtype=torch.long)
    #         }

    #     # Batch process all valid samples at once
    #     texts, images = zip(*valid_samples)
        
    #     try:
    #         inputs = self.processor(
    #             text=list(texts), 
    #             images=list(images), 
    #             return_tensors="pt", 
    #             padding=True, 
    #             max_length=self.max_length, 
    #             truncation=True
    #         )
    #     except Exception as e:
    #         # if self.rank == 0:
    #         print(f"Warning: Batch processing failed: {e}")
    #         return {
    #             "input_ids": torch.tensor([], dtype=torch.long),
    #             "attention_mask": torch.tensor([], dtype=torch.long),
    #             "labels": torch.tensor([], dtype=torch.long)
    #         }

    #     labels = inputs["input_ids"].clone()
        
    #     # Get image token ID once and reuse
    #     image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
    #         self.processor.tokenizer.special_tokens_map["boi_token"]
    #     )

    #     # Create combined mask for tokens to ignore in loss computation
    #     mask = (
    #         (labels == self.processor.tokenizer.pad_token_id) |
    #         (labels == image_token_id) |
    #         (labels == 262144)
    #     )
    #     # labels = torch.where(mask, torch.tensor(-100, dtype=labels.dtype), labels)
    #     labels[mask] = -100
    #     del mask
        
    #     inputs['labels'] = labels
        
    #     return inputs
    

    # For Pali-Gemma
    def format_data(self, image, question, answer):
        # PaliGemma format: <image> question\nanswer
        return {
            "image": image,
            "text": f"{question}\n{answer}"  # Simple concatenation
        }

    def collate_fn(self, batch):
        if self.processor is None:
            raise ValueError("Processor not initialized")

        images = []
        texts = []
        
        for example in batch:
            if example.get("image") is None:
                print("Warning: Skipping sample with missing image")
                continue
                
            images.append(example["image"])
            texts.append(example["text"])
        
        if not images:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)
            }
        
        # PaliGemma processor handles image tokenization differently
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # For PaliGemma, mask only padding tokens
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs['labels'] = labels
        return inputs


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
