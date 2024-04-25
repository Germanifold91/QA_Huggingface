# LOKA Assesment Test [German Romero]

# 1. Overview
The following project will present a Question Answering system designed to provide answers from multiple documents based on a user's input question. This repository is part of LOKA's technical assessment for the Machine Learning position. It is intended to showcase the candidate's approach to the problem.

# 2. Two-Step Approach
The problem in question tackled by this repo is divided in two stages: 
- **Text Search**:This feature enhances document search efficiency by using **embeddings** to represent text semantically within a high-dimensional space. Coupled with [**FAISS (Facebook AI Similarity Search)**](https://ai.meta.com/tools/faiss/) indexing, the system quickly identifies the most relevant documents to any query. Embeddings capture the core meaning of text, while FAISS enables fast, scalable retrieval by comparing these embeddings to find the closest matches. This combination ensures fast and precise search results even in large document databases. 

- **Question Answering**: This system leverages a **pre-trained question answering (QA) model**, upon receiving a user's question, the model scans through the documents selected during the semantic search process and hence identify those passages that potentially contain the answer. The process is powered by [Huggingface's](https://huggingface.co/) transformers library. Additionally the project allows to fine tune the model for better performance. This is subjected to the construction of an appropiate training dataset for the task. 

# 3. Project Structure 

## 3.1. General Flow

