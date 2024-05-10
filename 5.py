#Load API keys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pinecone import Pinecone
pc = Pinecone()
# delete index
index_name = 'langchain'
if index_name in pc.list_indexes().names():
    print(f'Deleting index {index_name} ....')
    pc.delete_index(index_name) 
    print('Done')

else:
    print(f'Index_name {index_name} doesnt exist!')