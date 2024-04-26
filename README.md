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
The script [`src/etl/data_processing.py`](src/etl/data_processing.py) encompasses all the processes responsible for converting raw files [documentation] into a format that is compatible for analysis by pre-trained models. Specifically, this set of functions performs the following tasks:
- **Transformation** of raw files from markdown to text files for model processing.
- **Construction** of dataset with metadata about documents.

### 3.2.2. Data Training
The process of fine-tuning a pre-trained model with new information requires that the training data adhere to a specific structure. The functions contained in [`src/etl/data_training.py`](src/etl/data_training.py) process the input data and output as a `Dataset` dataframe and a JSON file formatted to meet the requirements of various QA models. It is important to note that both of these functions accept input structured such that each question is paired with a corresponding context and an answer located within that text. Additionally, the creation of these context-question-answer triplets is typically labor-intensive.

```json

{
    "documents": [
        {
            "title": "sagemaker-projects",
            "contexts": [
                "SageMaker Projects help organizations set up and standardize developer environments for data scientists and CI/CD systems for MLOps engineers.",
                "No. SageMaker pipelines are standalone entities just like training jobs, processing jobs, and other SageMaker jobs. You can create, update, and run pipelines directly within a notebook by using the SageMaker Python SDK without using a SageMaker project.",
            ],
            "questions_answers": [
                {
                    "context_index": 0,
                    "question": "What is SageMaker",
                    "answer": "help organizations set up and standardize developer environments for data scientists"
                },
                {
                    "context_index": 1,
                    "question": "Is it neccesary to create a project in order to run a SageMaker pipeline?",
                    "answer": "No. SageMaker pipelines are standalone entities just like training jobs, processing jobs, and other SageMaker jobs."
                },
            ]
       }
   ]
}
```
## 3.3. Features

### 3.3.1. QA Class
The DocumentAssistant class, defined in the [`src/features/qa_class.py`](src/features/qa_class.py) module, is designed for document processing and question answering functionalities. This class utilizes the transformers library from Huggingface to encode text and generate embeddings that facilitate deep understanding and interactions with textual data.

**Key Features**
- **Tokenization and Modeling:** Utilizes AutoTokenizer for text encoding and AutoModel for generating text embeddings.
- **Document Processing and Search:** Implements methods to convert texts into embeddings and search these embeddings to find relevant texts based on a given question.
- **Question Answering:** Capable of answering questions using either the base model or a fine-tuned model, depending on the use case requirements. This is enhanced through a pipeline that integrates text searching with question answering.

### 3.3.2. QA Pipeline
The primary functionality of this project is encapsulated in the script [`src/features/qa_pipe.py`](src/features/qa_pipe.py). Executing this script initiates the main function which creates an instance of the DocumentAssistant class. This class is responsible for processing an input question and returning both the answer and the relevant documents associated with it. This setup ensures a seamless integration of text processing and retrieval functionalities to address user queries effectively.

**Sample Execution**
<p align="center">
<img width="952" alt="Screenshot 2024-04-25 at 9 56 35 PM" src="https://github.com/Germanifold91/loka_qa/assets/102771524/e1a030e7-66ad-4220-90c4-b0d7fa641e31">
</p>

## 3.4. Model Tracking
Upon execution of the question answering pipeline the function `track_execution()` part of [`src/model_tracking/tracking.py`](src/model_tracking/tracking.py) will save the input question as well as the outputs associated to the model such as the answer as well as:
- **Path to Relevant Document**
- **Score**
- **Start index of the answer**
- **End index of the answer**


> 💡 **This could be beneficial for identifying areas where efforts should be focused when developing future training datasets.**

**Sample Log**
```json
[
    {
        "question": "What are all AWS regions where SageMaker is available?",
        "answer": "East/West",
        "document_path": [
            "sagemaker-compliance.md"
        ],
        "score": 0.0060913050547242165,
        "start": 273,
        "end": 282
    }
]
```

## 3.5. Local File Configuration
The execution of the project requires to declare all the relevant paths and parameters in a yaml file located at [`conf/local.yml`](conf/local.yml), as follows:
```yaml
# config.yaml
data_processing_params:
  documents_path: /data/01_raw/md_files
  save_csv: False
  csvdf_path: /data/02_intermediate/tables/master_table.csv
  text_files_path: /data/02_intermediate/txt_files/
  arrowdf_path: /data/02_intermediate/tables/text_df

training_params:
  context_qa: /data/01_raw/training_objects/training_contexts.json
  training_json: /data/02_intermediate/training_data/training_data.json
  training_dataset: /data/02_intermediate/training_dataset

model_tracking_params:
  tracking_json_path: /data/04_model_output/tracking.json
```

# 4. System Execution

## 4.1. Clone Repo
``` bash
git clone https://github.com/Germanifold91/loka_qa
cd loka_qa
```


## 4.2. Setup
> 📝 **Note**: As of **2024/04/25** the [installation of FAISS must be through the use of conda](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for the appropiate functioning of this method. 

To set up the project environment to run the project, follow these steps:

```bash
make create-env
```
This command will create a Conda environment with all the necessary dependencies installed.


## 4.3. Running the Application

### 4.3.1. Data Processing
To run the data processing script, execute the following command:

```bash
make data
```

### 4.3.2. Running the QA pipeline
To run the question answering pipeline with a question as input, you can use:
```bash
make process-question QUESTION="Your question here"
```
### 4.3.3. Process Training Data
In order to generate the appropiate data for fine tunning a model run:
```bash
make data-training
```

### 4.3.4. Fine Tune Model
Fine Tunning of the model can be performed through:
```bash
make tune-model
```

# 5. Suggested Integration on AWS Architecture

This project utilizes AWS services to create a scalable, efficient, and robust system that can handle simultaneous requests, dynamically improve performance, and manage the creation of training datasets. Below is the revised architecture including the integration of Amazon SageMaker Ground Truth for dataset generation:

## 5.1. User Interaction
Users interact with the system via a web interface or an API. **Amazon API Gateway** serves as the gateway for all user requests, routing them to the appropriate AWS services.

## 5.2. Document Processing and Storage
- **AWS Lambda**: Lambda functions are initially triggered to process markdown documents upon upload or update. These functions read the documents from Amazon S3, convert them into structured text files, and store these processed files back in S3. This processing occurs only once for each document unless they are updated, ensuring efficiency in resource use.
- **Amazon S3**: Acts as the central repository for both the original markdown documents and the processed text files. This setup provides secure, durable, and scalable storage.
- 
## 5.3. Retrieving Processed Documents
- **AWS Lambda**: For answering user queries, Lambda functions are triggered by API Gateway. These functions quickly retrieve the pre-processed text files from S3 based on the user's request. Since the documents are already processed, the retrieval and response are fast and efficient.
The Lambda function uses the processed data to answer the user’s query using the Hugging Face model, ensuring that the responses are accurate and delivered promptly.

## 5.4. Logging User Queries and Responses
- **Amazon DynamoDB**: After generating a response to a user's question, details of the interaction, including the user's question, the answer, the score, and the start and end index of each answer within the document, are logged in a structured JSON format in DynamoDB.
- **AWS Lambda**: The same Lambda function that retrieves and processes the user’s query will also handle logging the details to DynamoDB. This integration ensures that every interaction is captured in real-time, allowing for detailed analytics and monitoring.

## 5.5. Training Dataset Generation with Amazon SageMaker Ground Truth
- **Amazon SageMaker Ground Truth**: To facilitate the fine-tuning of the Hugging Face model, Ground Truth is used to create accurate training datasets. This service allows you to employ human annotators, automated data labeling using machine learning, and a combination of both to generate high-quality datasets from your stored documents.
- Integration: Ground Truth can directly access documents stored in S3, utilize annotations and metadata logged in DynamoDB, and process data to generate datasets optimized for training machine learning models.

## 5.6. Model Fine-Tuning
- **AWS SageMaker**: Utilizes the datasets prepared by Ground Truth for training and fine-tuning the Hugging Face model. SageMaker provides the necessary computational resources and tools to manage the training process effectively.
- Model Updates: Once fine-tuning is complete, the updated model is deployed automatically to replace the older version, ensuring that the system benefits from continuous learning and improvement.

## 5.7. Scalability and Efficiency
- Automatic Scaling: AWS Lambda, DynamoDB, SageMaker, and Ground Truth automatically scale based on the demand. This feature ensures that the system can handle peak loads efficiently and adapt to increasing data and processing requirements without any manual intervention.
- Cost-Effectiveness: By automating the creation of training datasets and model fine-tuning, and leveraging serverless and managed services, the system remains cost-effective while continuously improving.





