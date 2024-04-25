""" This module prepares the training data for a question answering model. """
from typing import List, Dict, Union
from datasets import Dataset
import json
import yaml


def prepare_train_data(documents: List[Dict[str, Union[str, List[str], List[Dict[str, str]]]]], 
                       output_path: str) -> Dict[str, List[Dict[str, List[Dict[str, List[Dict[str, str]]]]]]]:
    """
    Prepares training data for a question answering model.

    Args:
        documents (List[Dict[str, Union[str, List[str], List[Dict[str, str]]]]]): A list of documents containing titles, contexts, and questions with answers.
        output_path (str): The path to save the prepared training data.

    Returns:
        Dict[str, List[Dict[str, List[Dict[str, List[Dict[str, str]]]]]]]: The prepared training data in a dictionary format.

    Raises:
        None
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
    

def load_and_format_data(data_file: str, save_dir: str) -> Dataset:
    """
    Loads data from a JSON file, formats it, and converts it to a Hugging Face Dataset.
    
    Args:
        data_file (str): The path to the JSON file containing the data.
        save_dir (str): The directory where the formatted dataset will be saved.
        
    Returns:
        Dataset: The formatted dataset.
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


if __name__ == "__main__":

    # Load the YAML configuration file
    with open("conf/local.yml", "r") as yamlfile:
        config = yaml.safe_load(yamlfile)  # Use safe_load to load the YAML configuration file safely

    # Load the training data from the configuration file
    data_training_params = config["training_params"]

    # Extract the documents from the loaded data
    with open(data_training_params["context_qa"], 'r') as f:
        data = json.load(f)

    # Prepare the training data
    documents = data["documents"]
    prepare_train_data(documents=documents, 
                       output_path=data_training_params["training_json"])
    
    # Prepare training dataset
    load_and_format_data(data_file=data_training_params["context_qa"],
                         save_dir=data_training_params["training_dataset"])


