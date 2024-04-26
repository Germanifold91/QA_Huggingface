""" The qa_class module provides the DocumentAssistant class for document processing and question answering. """

from transformers import AutoTokenizer, AutoModel, pipeline
from torch import Tensor
from datasets import Dataset
from typing import List, Any, Tuple, Dict
import torch
import os


class DocumentAssistant:
    """
    The DocumentAssistant class provides methods for document processing and question answering.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used for encoding text.
        model (AutoModel): The model used for generating text embeddings.

    Methods:
        mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
            Computes the mean pooling of the token embeddings.

        get_embeddings(text_list: List[str]) -> Tensor:
            Encodes a list of texts and returns their embeddings.

        text_search(master_dataset: Dataset, question: str, num_sources: int = 1) -> List[Any]:
            Searches for relevant texts in the master dataset based on a given question.

        question_answering(question: str, paths_list: List[str], fine_tuned_model: Dict[str, Any] = {"use_fine_tuned_model" : False, "model_path" : None}) -> Tuple[str, List[str]]:
            Performs question answering on a list of texts and returns the answer, file names, score, start, and end positions.

        answer_pipeline(master_dataset: Dataset, question: str) -> Tuple[str, List[str]]:
            Executes the text search and question answering pipeline and returns the answer and file names.

    """

    def __init__(
        self, model_ckpt: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

    def mean_pooling(self, model_output: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Computes the mean pooling of the token embeddings.

        This method takes the output tensor from the model and the attention mask tensor
        as input and computes the mean pooling of the token embeddings. Mean pooling is
        a technique used to summarize the information from the token embeddings by taking
        the average of all the embeddings.

        Args:
            model_output (Tensor): The output tensor from the model.
            attention_mask (Tensor): The attention mask tensor.

        Returns:
            Tensor: The mean-pooled embeddings.

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
        Retrieves the embeddings for a list of input texts.

        Args:
            text_list (List[str]): A list of input texts.

        Returns:
            Tensor: The embeddings of the input texts.

        """
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def text_search(
        self, master_dataset: Dataset, question: str, num_sources: int = 1
    ) -> List[Any]:
        """
        Searches for relevant texts in the master dataset based on a given question.

        Args:
            master_dataset (Dataset): The dataset containing the texts to search.
            question (str): The question to search for.
            num_sources (int): The number of relevant texts to retrieve (default: 1).

        Returns:
            List[Any]: The list of relevant texts.

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

    def question_answering(
        self,
        question: str,
        paths_list: List[str],
        fine_tuned_model: Dict[str, Any] = {
            "use_fine_tuned_model": False,
            "model_path": None,
        },
    ) -> Tuple[str, List[str], float, int, int]:
        """
        Performs question answering on a list of texts and returns the answer, file names, score, start, and end positions.

        Args:
            question (str): The question to answer.
            paths_list (List[str]): The list of file paths containing the texts to analyze.
            fine_tuned_model (Dict[str, Any]): The fine-tuned model configuration (default: {"use_fine_tuned_model" : False, "model_path" : None}).

        Returns:
            Tuple[str, List[str]]: The answer, file names, score, start position, and end position.

        """
        context = ""
        file_names = []

        for path in paths_list:
            with open(path, "r", encoding="utf-8") as file:
                context += "\n" + file.read()
            file_names.append(os.path.basename(path))

        if fine_tuned_model["use_fine_tuned_model"]:
            qa_pipeline = pipeline(
                "question-answering", model=fine_tuned_model["model_path"]
            )
        else:
            qa_pipeline = pipeline(
                "question-answering", model="deepset/roberta-base-squad2"
            )

        result = qa_pipeline(question=question, context=context, max_answer_len=100)

        return (
            result["answer"],
            file_names,
            result["score"],
            result["start"],
            result["end"],
        )

    def answer_pipeline(
        self, master_dataset: Dataset, question: str
    ) -> Tuple[str, List[str], float, int, int]:
        """
        Executes the text search and question answering pipeline and returns the answer and file names.

        Args:
            master_dataset (Dataset): The dataset containing the texts to search.
            question (str): The question to answer.

        Returns:
            Tuple[str, List[str]]: The answer, file names, score, start position, and end position.

        """
        texts_paths = self.text_search(master_dataset=master_dataset, question=question)

        answer, files_names, score, start, end = self.question_answering(
            question=question, paths_list=texts_paths
        )

        # Replace '.txt' with '.md' in the file names
        md_files_names = [fname.replace(".txt", ".md") for fname in files_names]

        return answer, md_files_names, score, start, end
