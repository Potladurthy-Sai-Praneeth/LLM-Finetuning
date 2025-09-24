from dotenv import load_dotenv
import os


# System message for the assistant
system_message = "You are an expert in Medical Diagnosis and treatment. You are given a image. You need to provide a description of the disease or anamoly or sometimes suggest possible treatments analysing the image."

SYSTEM_MESSAGE_CONTENT = [{"type": "text", "text": system_message}]
    
def format_data(sample):
    '''
    This function returns a dictionary formatted for training/fine-tuning a multimodal model.
    '''
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE_CONTENT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample['question']},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ],
    }


def process_dataset(dataset):
    '''
    This function processes the dataset to create a list of formatted samples.
    Each sample contains an image, a question, and an answer.
    Note: dataset.map is not used as it serializes the image object which is not desired.
    '''
    processed_data = []

    for point in dataset:
        image,questions,answers = point['image'],point['questions'],point['answers']
        for pair in zip(questions,answers):
            question,answer = pair
            sample = {
                "image": image,
                "question": question,
                "answer": answer
            }
            formatted_sample = format_data(sample)
            processed_data.append(formatted_sample)
    
    return processed_data


def collate_fn(batch,processor):
    texts = []
    images = []
    for example in batch:
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(example['messages'][1]['content'][1]['image'])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    batch['labels'] = batch["input_ids"].clone()
    
    return batch


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

