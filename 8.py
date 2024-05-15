# PROJECT Transform Loaders for a Q&A system using data from own knowledge base (beyond LLM trained data) by GPT model inputs using "embedded-based search"

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

step = 7

#Loading PDFs
if step==1:
    def load_document(file):
#        from langchain.document_loaders import PyPDFLoader *depracated*
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data

    #running code    
    data = load_document('files/ML_predictive_modeling_diabetes_2023_DMSO.pdf')
#    data = load_document(http://xxxxxxx-ML_predictive_modeling_diabetes_2023_DMSO.pdf')

#    print(data[2].page_content)
#    print(data[1].metadata) # metadata argument and page
#    print(f'you have {len(data)} pages in your PDF')
    print(f'you have {len(data[1]).page_content} characters in the page')


# Loader that discovers different file formats
if step==2:

#    pip install docx2txt

    def load_document(file):
        name, extension = os.path.splitext(file)

    
        if extension == '.pdf':
#            from langchain.document_loaders import PyPDFLoader *depracated*
            from langchain_community.document_loaders import PyPDFLoader
            print(f'Loading{file}')
            loader = PyPDFLoader(file)

        elif extension == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            print(f'Loading{file}')
            loader = Docx2txtLoader(file)

        else:
            print('unsupported format')
            return None

        data = loader.load()
        return data
    
    #running code
    data = load_document('files/GreatDocx.docx')
#    data = load_document('files/ML_predictive_modeling_diabetes_2023_DMSO.pdf')

    print(data[0].page_content)
    print(data[0].metadata) # metadata argument and page
    print(f'you have {len(data)} pages in your file')
    print(f'you have {len(data[0]).page_content} characters in the page')



# Loader from online services 
if step==3:

#    pip install wikipedia

    def load_wikipedia(query, lang='en', load_max_docs=2):
        from langchain_community.document_loaders import WikipediaLoader
        loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
        data = loader.load()
        return data
    
    
    #running code
#    data = load_wikipedia('GPT-4')
    data = load_wikipedia('JayZ')

    print(data[0].page_content)
    print(data[0].metadata) # metadata argument and page
    print(f'you have {len(data)} pages in your file')
    print(f'you have {len(data[0]).page_content} characters in the page')



# Chunk embedding to optimizing content for vector database
if step==4:

    #chunk embedding
    def chunk_data(data, chunk_size=256):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_documents(data)
        return chunks
    
    # pdf loader
    def load_document(file):
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data

    #OpenAI embedding cost estimator
    def print_embedding_cost(texts):
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        print(f'Total tokens: {total_tokens}')
        print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


    #running code    
    data = load_document('files/ML_predictive_modeling_diabetes_2023_DMSO.pdf')
    chunks=chunk_data(data)
    print(len(chunks)) #show how many chunks of data
    print(chunks[2].page_content) #show content of [index] chunk
    print_embedding_cost(chunks)


    # Embedding and uploading to vector DB (pinecone)
if step==5:
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()

    #chunk embedding
    def chunk_data(data, chunk_size=256):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_documents(data)
        return chunks
    
    # pdf loader
    def load_document(file):
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data

#Create some functions for chunk + embedding 
    def insert_or_fetch_embeddings(index_name, chunks): #creates vector DB index if index is new, embeds chunk and adds both to pinecone else just loads embeddings from that index

        #NB: pinecone is the library, Pinecone is a class in the pinecone library
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

        if index_name in pc.list_indexes().names():
            print(f'Index {index_name} already exists. Load embeddings ...', end='')
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            print('OK')

        else:
            print(f'Creating index {index_name} and embeddings ... ', end='')
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=PodSpec(environment='gcp-starter')
            )
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
            # process chunks, generates embeddings, inserting index to pinecone and return the vecor store
            print('ok')
            return vector_store
        

    data = load_document('files/ML_predictive_modeling_diabetes_2023_DMSO.pdf')
    chunks=chunk_data(data)

    index_name = 'index1'
    vector_store = insert_or_fetch_embeddings(index_name, chunks)


if step==6:

    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()

# Index maintenance function
    def delete_pinecone_index(index_name='all'):
        if index_name == 'all':
            indexes = pc.list_indexes().names()
            print('deleting all indexes ...')
            for index in indexes:
                pc.delete_index(index)
            print('OK')
        else:
            print(f'Deleting index {index_name} ... ', end='')
            pc.delete_index(index_name)
            print('OK')


    delete_pinecone_index('all')


if step==7:
#asking and getting answers

    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()

    #chunk embedding
    def chunk_data(data, chunk_size=256):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_documents(data)
        return chunks
    
    # pdf loader
    def load_document(file):
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data

#Create some functions for chunk + embedding 
    def insert_or_fetch_embeddings(index_name, chunks): #creates vector DB index if index is new, embeds chunk and adds both to pinecone else just loads embeddings from that index

        #NB: pinecone is the library, Pinecone is a class in the pinecone library
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

        if index_name in pc.list_indexes().names():
            print(f'Index {index_name} already exists. Load embeddings ...', end='')
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            print('OK')

        else:
            print(f'Creating index {index_name} and embeddings ... ', end='')
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=PodSpec(environment='gcp-starter')
            )
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
            # process chunks, generates embeddings, inserting index to pinecone and return the vecor store
            print('ok')
            return vector_store


    data = load_document('files/ML_predictive_modeling_diabetes_2023_DMSO.pdf')
    chunks=chunk_data(data)

    index_name = 'index1'
    vector_store = insert_or_fetch_embeddings(index_name, chunks)


    def QAfunction(vector_store, question):
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        answer = chain.invoke(question)
        return answer

    question = 'what is the document about?'
    answer = QAfunction(vector_store, question)
    print(answer)    



if step==8:
    import time
    i = 1
    print('Type a question or enter Quit or Exit')
    while True:
        question = input(f'Question #{1}')
        i=i+1
        if question.lower() in ['quit', 'exit']:
            print('Quitting ...bye')
            time.sleep(2)
            break
        answer = QAfunction(vector_store, question)
        print(f'\nAnswer: {answer}')
        print(f'\n {"-" * 50} \n')


if step==9:

    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    def load_wikipedia(query, lang='en', load_max_docs=2):
        from langchain_community.document_loaders import WikipediaLoader
        loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
        data = loader.load()
        return data


    #chunk embedding
    def chunk_data(data, chunk_size=256):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_documents(data)
        return chunks


#Create some functions for chunk + embedding 
    def insert_or_fetch_embeddings(index_name, chunks): #creates vector DB index if index is new, embeds chunk and adds both to pinecone else just loads embeddings from that index

        #NB: pinecone is the library, Pinecone is a class in the pinecone library
#        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

        if index_name in pc.list_indexes().names():
            print(f'Index {index_name} already exists. Load embeddings ...', end='')
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            print('OK')

        else:
            print(f'Creating index {index_name} and embeddings ... ', end='')
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=PodSpec(environment='gcp-starter')
            )
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
            # process chunks, generates embeddings, inserting index to pinecone and return the vecor store
            print('ok')
            return vector_store


    def QAfunction(vector_store, question):
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        answer = chain.invoke(question)
        return answer


    
    #running code
    data = load_wikipedia('JayZ')
    chunks = chunk_data(data)
    index_name = 'jayz'
    vector_store = insert_or_fetch_embeddings(index_name, chunks=chunks)
    question = 'who is jayz'
    answer = QAfunction(vector_store, question)
    print(answer)
    

