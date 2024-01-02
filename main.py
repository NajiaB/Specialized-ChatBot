# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 14:50:07 2024

@author: BOUADDOUCH Najia
"""

import streamlit as st
import PyPDF2 as pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import tiktoken
OPENAI_API_KEY = "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"

st.header('2022 World Cup chatbot')

with st.sidebar:
    st.title("documents")
    file = st.file_uploader("please upload a PDF", type="pdf")
    
if file is not None:
    pdf_reader = pdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text = text + page.extract_text()
        #st.write(text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len)
    chunks = text_splitter.split_text(text)
    #st.write(chunks)
    
    #embeddings :


    embd = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    #VECTOR STORE WITH FAISS
    
    vector_store =FAISS.from_texts(chunks, embd) #generarte embedding + initialize VECTORDB faiss + store chunks and embd
    
    # user input
    
    question = st.text_input("enter question")
    
    #similarity search
    if question :
        match=vector_store.similarity_search(question)
        #st.write(match)
        
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                        temperature =0,
                        max_tokens=1000
                        )
        
        chain = load_qa_chain(llm,chain_type="stuff")
        response= chain.run(input_documents=match, question=question)
        st.write(response)
        
        
        
    
    
     
    
