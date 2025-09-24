from dotenv import load_dotenv
import os
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self,dataset) -> None:
        super().__init__()

        self.dataset = dataset
        self.system_message = "You are an expert in Medical Diagnosis and treatment. You are given a image. You need to provide a description of the disease or anamoly or sometimes suggest possible treatments analysing the image."
        self.system_message_content = [{"type": "text", "text": self.system_message}]

        self.index_mapping = []

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

        return self.format_data(image,question,answer)


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

