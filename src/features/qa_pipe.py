""" This script processes a dataset and a question using the DocumentAssistant model. """
from src.features.qa_class import DocumentAssistant
from src.model_tracking.tracking import track_execution  
from argparse import ArgumentParser
from datasets import Dataset, load_from_disk
from typing import List, Tuple
import yaml

def main(dataset: Dataset, 
         question: str) -> Tuple[str, List[str]]:
    """
    Process the given dataset and question using the DocumentAssistant model.
    
    Args:
        dataset (Dataset): The dataset to be processed.
        question (str): The question to be answered.
    
    Returns:
        None
    """
    doc_assistant = DocumentAssistant(model_ckpt="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    
    # Use the DocumentAssistant to process the question
    answer, files_names, score, start, end  = doc_assistant.answer_pipeline(master_dataset=dataset, question=question)
    
    # Print the answer and the names of relevant files
    print("Answer:", answer)
    print("Relevant document(s):", files_names)
    print("Score:", score)

    return answer, files_names, score, start, end

if __name__ == "__main__":
        
    # Load the YAML configuration file
    with open("conf/local.yml", "r") as yamlfile:
        config = yaml.safe_load(yamlfile)  # Use safe_load to load the YAML configuration file safely

    # Parse command-line arguments
    parser = ArgumentParser(description="Answer questions based on a dataset using DocumentAssistant.")
    parser.add_argument('--question', type=str, default=None, help="The question you want answered. Leave empty to prompt.")
    
    args = parser.parse_args()

    # If no question was provided via command-line arguments, prompt the user
    if args.question is None:
        args.question = input("Please write your question here: ")

    # Load the dataset from disk
    dataset_path = config["data_processing_params"]["arrowdf_path"]
    master_dataset = load_from_disk(dataset_path)

    # Process the dataset and question
    answer, files, score, start, end = main(dataset=master_dataset, question=args.question)
    output_dict = {"question": args.question, 
                   "answer": answer, 
                   "document_path": files,
                   "score": score,
                   "start": start,
                   "end": end}
    
    json_tracking_path = config["model_tracking_params"]["tracking_json_path"]
    track_execution(output_dict, json_tracking_path)