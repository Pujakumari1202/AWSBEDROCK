import json
import os
import sys
import boto3
import streamlit as st


## we Will be using Titan Embedding for this model to generate embeddings
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock



## Data Ingection
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


# vector embeddings and Vectors store
from langchain_community.vectorstores import FAISS


## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import  RetrievalQA


##Bedrock clients(created the client)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



#Data Ingestion
def data_ingection():
    ## reading the data
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # in our testing character split works better with this pdf data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)



    docs=text_splitter.split_documents(documents)
    return docs

## vector Embeddings and Vectors store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    # store in faiss_store database
    vectorstore_faiss.save_local("faiss_index")


# now we will use some LLM models
def get_clude_llm():
    ## create the Anthropic model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,model_kwargs={'maxTokens':512})


    return llm



def get_llama3_llm():
    ## create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0"
,client=bedrock,model_kwargs={'maxTokens':512})


    return llm





## now create prompt template(we are using langchain)
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

## langchain prompt template we are using
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])



## response 
def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    ## we are doing similarity search from vectorstore_faiss
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity",search_kwargs={"k":3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    
    # retrievr the ans and stored in answer variable
    answer=qa({"query":query})
    # in answer there is result variable which has the entire ans
    return answer['result']


## Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")  

    st.header("Chat with PDF using AWS Bedrock💁")


    user_question=st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or create Vector Store:")

        if st.button("Vector Update"):
            with st.spinner("Processing..."):
                # Calling the functions
                docs=data_ingection()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing...."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_clude_llm()


            st.write(get_response_llm(llm,faiss_index,user_question)
                    )
            st.success("Done")


    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llama3_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


if __name__=="__main__":
    main()



