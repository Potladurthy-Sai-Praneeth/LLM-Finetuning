import os
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self,dataset,processor,img_size=(448,448),max_length=256) -> None:
        super().__init__()

        self.dataset = dataset
        self.processor = processor
        self.system_message = "You are an expert in Medical Diagnosis and treatment. You are given a image. You need to provide a description of the disease or anamoly or sometimes suggest possible treatments analysing the image."
        self.system_message_content = [{"type": "text", "text": self.system_message}]

        self.index_mapping = []
        self.img_size = int(img_size[0]),int(img_size[1])
        self.max_length = int(max_length)

        for sample_idx, point in enumerate(self.dataset):
            questions = point['questions']
            for qa_idx in range(len(questions)):
                self.index_mapping.append((sample_idx, qa_idx))
        
    def __len__(self):
        return len(self.index_mapping)
    
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

    def __getitem__(self, idx):
        sample_idx, qa_idx = self.index_mapping[idx]
        point = self.dataset[sample_idx]
        question = point['questions'][qa_idx]
        answer = point['answers'][qa_idx]
        image = point['image']        
        image = image.resize(self.img_size)
        
        return self.format_data(image,question,answer)
    

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
                # if self.rank == 0:
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
                max_length=self.max_length, 
                truncation=True
            )
        except Exception as e:
            # if self.rank == 0:
            print(f"Warning: Batch processing failed: {e}")
            return {}

        # Create labels efficiently using in-place operations where possible
        labels = inputs["input_ids"].clone()
        image_token_id = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])]
        image_token_mask = (labels == image_token_id)
        pad_token_mask = (labels == self.processor.tokenizer.pad_token_id)
        # attention_mask_zero = (inputs['attention_mask'] == 0)
        special_token_mask = (labels == 262144)
        
        labels[pad_token_mask | image_token_mask | special_token_mask] = -100
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





# if __name__ == "__main__":
#     load_dotenv()
#     DATASET_ID = os.getenv("DATASET_ID")
    
#     print(f"Loading dataset: {DATASET_ID}")
#     dataset = load_dataset(DATASET_ID, split="train")
#     dataset = dataset.select(50)

#     print(f"Dataset loaded with {len(dataset)} samples")

#     processed_data = process_dataset(dataset)

#     print(f"Processed {len(processed_data)} samples.")
#     print("Sample processed data:")
#     print(processed_data[0])

