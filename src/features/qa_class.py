from transformers import AutoTokenizer, AutoModel, pipeline
from torch import Tensor
from datasets import Dataset
from typing import List, Any, Tuple
import torch
import os


class DocumentAssistant:
    def __init__(self, model_ckpt: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

    def mean_pooling(self, model_output: Tensor, attention_mask: Tensor) -> Tensor:
        """

        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, text_list: List[str]) -> Tensor:
        """

        """
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        return self.mean_pooling(model_output, encoded_input["attention_mask"])
    
    def text_search(self, master_dataset: Dataset, question: str, num_sources: int = 1) -> List[Any]:
        """

        """
        embedded_dataset = master_dataset.map(
            lambda x: {"EMBEDDINGS": self.get_embeddings([x["TEXT"]])[0].cpu().numpy()}
        )

        embedded_dataset.add_faiss_index(column="EMBEDDINGS")
        question_embedding = self.get_embeddings([question])[0].cpu().detach().numpy()

        _, samples = embedded_dataset.get_nearest_examples(
            "EMBEDDINGS", question_embedding, k=num_sources
        )

        return samples["TEXT_FILE_PATH"]
    
    def question_answering(self, question: str, paths_list: List[str]) -> Tuple[str, List[str]]:
        """

        """
        context = ""
        file_names = []

        for path in paths_list:
            with open(path, 'r', encoding='utf-8') as file:
                context += "\n" + file.read()
            file_names.append(os.path.basename(path))

        qa_pipeline = pipeline("question-answering", model = "deepset/roberta-base-squad2")
        
        result = qa_pipeline(question=question, context=context, max_answer_len=100)
        
        return result['answer'], file_names, result['score'], result['start'], result['end']
    
    def answer_pipeline(self, master_dataset: Dataset, question: str) -> Tuple[str, List[str]]:
        """

        """
        texts_paths = self.text_search(master_dataset=master_dataset, question=question)
        
        answer, files_names, score, start, end = self.question_answering(question=question, paths_list=texts_paths)

        # Replace '.txt' with '.md' in the file names
        md_files_names = [fname.replace('.txt', '.md') for fname in files_names]
        
        return answer, md_files_names, score, start, end
