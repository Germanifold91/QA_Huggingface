from typing import List, Dict, Union
from datasets import Dataset
import json


def prepare_train_data(documents: List[Dict[str, Union[str, List[str], List[Dict[str, str]]]]], 
                       output_path: str) -> Dict[str, List[Dict[str, List[Dict[str, List[Dict[str, str]]]]]]]:
    """
    """
    data = []
    question_id = 1  # Initialize question ID counter

    for document in documents:
        title = document["title"]
        contexts = document["contexts"]
        questions_answers = document["questions_answers"]
        
        paragraphs = []
        for i, context in enumerate(contexts):
            qas = []
            for qa in questions_answers:
                if qa["context_index"] == i:
                    answer_start = context.find(qa["answer"])
                    if answer_start != -1:
                        qas.append({
                            "id": str(question_id).zfill(5),  # Use the question ID counter
                            "is_impossible": False,
                            "question": qa["question"],
                            "answers": [
                                {
                                    "text": qa["answer"],
                                    "answer_start": answer_start,
                                }
                            ],
                        })
                        question_id += 1  # Increment the question ID counter
            paragraphs.append({
                "context": context,
                "qas": qas,
            })
        
        data.append({
            "title": title,
            "paragraphs": paragraphs,
        })

    train_data = {"data": data}
    
    with open(output_path, 'w') as f:
        json.dump(train_data, f)
    
    return train_data

def load_and_format_data(data_file: str, save_dir: str) -> Dataset:
    """
    """
    # Load the data from a JSON file
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Prepare the data in the format required for fine-tuning
    formatted_data = {
        'title': [],
        'context': [],
        'question': [],
        'answers': [],
    }
    for document in data['documents']:
        title = document['title']
        for qa in document['questions_answers']:
            context = document['contexts'][qa['context_index']]
            formatted_data['title'].append(title)
            formatted_data['context'].append(context)
            formatted_data['question'].append(qa['question'])
            formatted_data['answers'].append({
                'text': [qa['answer']],
                'answer_start': [context.find(qa['answer'])],
            })

    # Convert the data to a Hugging Face Dataset
    dataset = Dataset.from_dict(formatted_data)

    # Save the dataset to a directory
    dataset.save_to_disk(save_dir)

    return dataset

if __name__ == "__main__":
    # Load the training data
    with open("data/01_raw/training_objects/training_contexts.json", 'r') as f:
        data = json.load(f)

    # Extract the documents from the loaded data
    documents = data["documents"]

    # Prepare the training data
    train_data = prepare_train_data(documents=documents, 
                                    output_path="data/02_intermediate/training_data/train_data.json")
    
    # Load and format the training data
    dataset = load_and_format_data()


