#Load API keys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import pinecone
from pinecone import Pinecone
pc = Pinecone()

#now create a new index
#import serverless spec from pinecone
from pinecone import ServerlessSpec
index_name = 'langchain'
if index_name not in pc.list_indexes().names():
    print(f'Creating index: {index_name}')
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print('index created!')
else:
    print(f'An index named {index_name} already exists!')


#get description of all indexes in pinecone project
pc.list_indexes()
#get index by index position 
pc.list_indexes()[0]
#get index by name 
#pc.describe_index('langchain')
#pc.list_indexes().names()


#operation on indexes
# select an index
index_name = 'langchain'
index = pc.Index('index_name')
index.describe_index_stats()