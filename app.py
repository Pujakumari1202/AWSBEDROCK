import json
import os
import sys
import boto3



## we Will be using Titan Embedding for this model to generate embeddings

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


## Data Ingection
import numpy as np
from langchain.document_loaders import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader


# vector embeddings and Vectors store
from langchain.vectorstores import FAISS


## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import  RetrievalQA


##Bedrock clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings

