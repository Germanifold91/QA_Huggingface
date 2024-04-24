from typing import List, Dict, Union
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

if __name__ == "__main__":
    # Load the training data
    with open("data/01_raw/training_objects/training_contexts.json", 'r') as f:
        data = json.load(f)

    # Extract the documents from the loaded data
    documents = data["documents"]

    # Prepare the training data
    train_data = prepare_train_data(documents=documents, 
                                    output_path="data/02_intermediate/training_data/train_data.json")
    



