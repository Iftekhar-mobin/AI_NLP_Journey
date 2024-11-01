# RAG Implementation with LangChain and Hugging Face

This project demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** model using LangChain and Hugging Face libraries. RAG combines document retrieval with a sequence-to-sequence model to generate answers based on retrieved documents. This implementation leverages pretrained dense retrieval models and language models to perform text embeddings, document chunking, and answer generation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Documentation](#documentation)
- [Challenges](#challenges)
- [Contributions](#contributions)
- [License](#license)

## Introduction

RAG models retrieve relevant documents from a database and pass them to a sequence-to-sequence model, generating responses based on both pretrained knowledge and the provided context. This setup combines document retrieval using FAISS with a question-answering pipeline using Hugging Face's transformer models.

In this repository, we demonstrate:
- Loading documents using Hugging Face datasets
- Document splitting with LangChain's `RecursiveCharacterTextSplitter`
- Text embedding and storage with FAISS vector store
- RAG workflow using `RetrievalQA` chain for question-answering

## Features

- **Document Loading**: Uses Hugging Face datasets to load instruction-based records.
- **Text Embedding**: Transforms text into embeddings using Hugging Face models for similarity search.
- **Vector Store**: Manages embeddings and allows efficient search with FAISS.
- **Retrieval & Generation**: Combines retrieval and generation to answer questions with relevant context.


## Documentation

### Key Libraries

- **LangChain**: Used for document loaders, text splitting, embeddings, and QA chains.
- **Hugging Face**: Provides pre-trained models for embeddings and QA pipelines.
- **FAISS**: Manages vector stores, enabling similarity search for efficient retrieval.

### Pipeline Structure

1. **Document Loading**: Load and prepare documents using LangChain’s loaders.
2. **Text Splitting**: Split documents to match the model's input size.
3. **Embedding**: Generate embeddings with Hugging Face models.
4. **Vector Storage**: Store embeddings in FAISS for fast similarity search.
5. **RetrievalQA**: Retrieve documents and generate answers with LangChain's `RetrievalQA` chain.

This consolidated script will:

- Install necessary libraries.
- Load documents from Hugging Face's dataset.
- Split the documents for processing.
- Embed the text for similarity search.
- Set up a retrieval and question-answering pipeline.
- Retrieve relevant data and generate an answer for a sample query.

## Setup

### Requirements

Install the necessary libraries:
```bash
pip install langchain torch transformers sentence-transformers datasets faiss-cpu
!pip install -q langchain
!pip install -q torch
!pip install -q transformers
!pip install -q sentence-transformers
!pip install -q datasets
!pip install -q faiss-gpu

# Install necessary libraries
!pip install langchain torch transformers sentence-transformers datasets faiss-cpu

This structure makes it easy to follow each step of the RAG workflow in one code block.

# Import libraries
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain.chains import RetrievalQA

# Step 1: Load Documents
loader = HuggingFaceDatasetLoader(dataset_name="databricks/databricks-dolly-15k", page_content_column="context")
data = loader.load()

# Step 2: Document Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# Step 3: Embedding & Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
db = FAISS.from_documents(docs, embeddings)

# Step 4: Prepare LLM Model for Question Answering
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Step 5: Retrieve and Generate Answer with RetrievalQA Chain
qa = RetrievalQA.from_chain_type(llm=question_answerer, chain_type="refine", retriever=db.as_retriever())
result = qa.run({"query": "Who is Thomas Jefferson?"})

# Display Result
print(result)

