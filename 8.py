# PROJECT Q&A system using data from own knowledge base (beyond LLM trained data) by GPT model inputs using "embedded-based search"

#   Typical pipeline as follows:
# 1. Prep the document (once per doc). This prepares the data being searched as chunk embeddings
#   a. Load the data into Langchain Documents
#   b. Embed chinks into numeric vectors (using embedding mode eg OpenAI text embedding agent)
#   c. Save chunks and embeddings to PineCone vector db
#
# 2. Search (once per query)
#   a. Embed the users questions Uwith same embedding model)
#   b. Using questions embedding and chunks embedding rank the vectors by similarity to questions embedding. 
#   get nearest vector 
#
# 3. Ask
#   a. insert question and most releveant chunks into msg to a GPT omdel
#   b. return GPTs answer
# 

#Load your OpenAI API key from .env file
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#pip install pypdf

step = 1

if step==1:
    def load_document(file):
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data
    
    data = load_document('files/ML_predictive_modelling_diabetes_2023_DMSO.pdf')
    print(data[1].page_content)
    print(data[10].metadata)
    
                         

