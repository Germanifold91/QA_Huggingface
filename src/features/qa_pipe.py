from src.features.qa_class import DocumentAssistant  
from datasets import Dataset


def main(dataset: Dataset, 
         question: str):
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
    answer, files_names = doc_assistant.answer_pipeline(master_dataset=dataset, question=question)
    
    # Print the answer and the names of relevant files
    print("Answer:", answer)
    print("Relevant document(s):", files_names)

    return answer, files_names