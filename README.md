# LOKA Assesment Test [German Romero]

# 1. Overview
The following project will present a Question Answering system designed to provide answers from multiple documents based on a user's input question. This repository is part of LOKA's technical assessment for the Machine Learning position. It is intended to showcase the candidate's approach to the problem.

# 2. Two-Step Approach
The problem in question tackled by this repo is divided in two stages: 
- **Text Search**:This feature enhances document search efficiency by using **embeddings** to represent text semantically within a high-dimensional space. Coupled with [**FAISS (Facebook AI Similarity Search)**](https://ai.meta.com/tools/faiss/) indexing, the system quickly identifies the most relevant documents to any query. Embeddings capture the core meaning of text, while FAISS enables fast, scalable retrieval by comparing these embeddings to find the closest matches. This combination ensures fast and precise search results even in large document databases. 

- **Question Answering**: This system leverages a **pre-trained question answering (QA) model**, upon receiving a user's question, the model scans through the documents selected during the semantic search process and hence identify those passages that potentially contain the answer. The process is powered by [Huggingface's](https://huggingface.co/) transformers library. Additionally the project allows to fine tune the model for better performance. This is subjected to the construction of an appropiate training dataset for the task. 

# 3. Project Structure 

## 3.1. General Flow

<p align="center">
<img width="533" alt="Pipeline LOKA" src="https://github.com/Germanifold91/loka_qa/assets/102771524/bc96830d-ab0d-4be4-846e-9fa8d12d0818"> 
</p>

## 3.2. ETL
### 3.2.1. Data Processing
The script `src/etl/data_processing.py` encompasses all the processes responsible for converting raw files [documentation] into a format that is compatible for analysis by pre-trained models. Specifically, this set of functions performs the following tasks:
- **Transformation** of raw files from markdown to text files for model processing.
- **Construction** of dataset with metadata about documents.
### 3.2.2. Data Training

The process of fine-tuning a pre-trained model with new information requires that the training data adhere to a specific structure. The functions contained in `src/etl/data_training.py` process the input data and output both a `Dataset` dataframe and a JSON file formatted to meet the requirements of various QA models. It is important to note that both of these functions accept input structured such that each question is paired with a corresponding context and an answer located within that text. Additionally, the creation of these context-question-answer triplets is typically labor-intensive.

```json

{
    "documents": [
        {
            "title": "France",
            "contexts": [
                "France, officially known as the French Republic, is a country located in Western Europe. It is known for its rich history, culture, and cuisine.",
                "Paris is the capital city of France and is famous for its iconic landmarks such as the Eiffel Tower and the Louvre Museum.",
                "French is the official language of France and is spoken by the majority of the population."
            ],
            "questions_answers": [
                {
                    "context_index": 0,
                    "question": "Where is France located?",
                    "answer": "a country located in Western Europe"
                },
                {
                    "context_index": 2,
                    "question": "What language is spoken by the majority of the population in France?",
                    "answer": "spoken by the majority of the population"
                },
            ]
       }
   ]
}
```
## 3.3. Features
### 3.3.1. QA Class
The DocumentAssistant class, defined in the `src/feature/qa_class.py module, is designed for document processing and question answering functionalities. This class utilizes the transformers library from Huggingface to encode text and generate embeddings that facilitate deep understanding and interactions with textual data.

**Key Features**
- **Tokenization and Modeling:** Utilizes AutoTokenizer for text encoding and AutoModel for generating text embeddings, ensuring that the textual data is processed with high accuracy.
- **Document Processing and Search:** Implements methods to convert texts into embeddings and search these embeddings to find relevant texts based on a given question.
- **Question Answering:** Capable of answering questions using either the base model or a fine-tuned model, depending on the use case requirements. This is enhanced through an intelligent pipeline that integrates text searching with question answering.
