#build front end QA app using streamlit

#Load your OpenAI API key from .env file

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pinecone
from pinecone import PodSpec
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings

# Loader that discovers different file formats
def load_document(file):
    name , extension = os.path.splitext(file)
    if extension == '.pdf':
    #    from langchain.document_loaders import PyPDFLoader *depracated*
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading{file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f'Loading{file}')
        loader = TextLoader(file)
    else:
        print('unsupported format')
        return None

    data = loader.load()
    return data


# chunk embedding
def chunk_data(data , chunk_size=256 , chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


#def create_embeddings(chunks):
#    embeddings = OpenAIEmbeddings()
#    vector_store = Chroma.from_documents(chunks, embeddings)
#    return vector_store


def create_embeddings(index_name, chunks):
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small' , dimensions=1536)

    print(f'Creating index {index_name} and embeddings ... ', end='')
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=PodSpec(environment='gcp-starter')
    )
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    print('ok')
    return vector_store


def QAfunction(vector_store , question , k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo' , temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity' , search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm , chain_type="stuff" , retriever=retriever)
    answer = chain.invoke(question)
    return answer


def calc_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens , total_tokens / 1000 * 0.0004


#delete chat history (using callback)
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv , find_dotenv

    load_dotenv(find_dotenv() , override=True)

    pc.delete_index('index0')

    # st.image('img.png')
    st.subheader('LLM QA app')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key' , type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        #Chunk size and k widget
        uploaded_file = st.file_uploader('Upload a file:' , type=['pdf' , 'docs' , 'txt'])
        chunk_size = st.number_input('Chunk size:' , min_value=1 , max_value=2048 , value=512, on_change=clear_history)
        k = st.number_input('k' , min_value=1 , max_value=20 , value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)
        # must refresh on screen chat history when new doc content is uploaded - we use callback function 'clear_history'

        if uploaded_file and add_data:
            with st.spinner('Read, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./' , uploaded_file.name)
                with open(file_name , 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data , chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, chunks: {len(chunks)}')

                tokens , embedding_cost = calc_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')


                #vector_store = create_embeddings(chunks=chunks)
                embeddings = OpenAIEmbeddings(model='text-embedding-3-small' , dimensions=1536)
                vector_store = create_embeddings(index_name='index0', chunks=chunks)

                st.session_state.vs = vector_store
                st.success('Job successful')

#add prompt box
    question = st.text_input('Ask something ...')
    if question:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = QAfunction(vector_store, question, k)
            st.text_area('Answer: ', value=answer)


#collect and display chat history
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {question} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat history:', value=h, key='history', height=400)



