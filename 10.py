# Chat model Summarization

import os
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv() , override=True)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0 , model_name='gpt-3.5-turbo')


step = 8


text = """
Sample text about Mojo programming language
Mojo Programming Language For Next Generation AI Developers 2024 May 6, 2024 by Ubaid Qureshi
Mojo Programming Language is a new and high-level programming language created by Chris Lattner, mastermind of Swift language. It started working in May 2019 and finally launched in early May 2023. The main goal of the Mojo language is to speed up code execution. Modulor launched its Mojo SDK 0.7 version on 25 January 2024. Mojo is a programming language that has gained popularity recently for its simplicity and versatility. Developed by a team of experienced programmers, Mojo was created to make it easier for developers to write code that is readable, maintainable, and efficient.
Mojo Programming Language
Mojo is a High-level programming language that combines Python syntax, metaprogramming, and system programming like C++. In this language most features are Python but some new features are also introduced that speed up the performance of the machine 35000x time compared to Python.
Mojo programming language release date
Mojo language was first launched on 2 May 2023. The founder or CEO of Mojo language is Charis Lattner. Charis Lattner started working on the Mojo language in 2019 and finally launched it on 2 May 2023. It is available on a web-based in May and it is locally introduced in LinuxOS in early September 2023.
"""



# summarizing a direct simple text input
if step==1:

    from langchain.schema import(
        AIMessage,
        HumanMessage,
        SystemMessage
        )


    messages = [
        SystemMessage(content='You are an expert copywriter with expertise in summarising documents'),
        HumanMessage(content=f'Please summarize a short, concise summary of the following text: \n TEXT: {text}')
    ]

    llm.get_num_tokens(text)
    summary_output = llm(messages)
    print(summary_output.content)



#DSummarizing using prompt template. But still subject to the token limits of the model
if step==2:

    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain


    template = '''
    Write a short, concise summary of the following text: \n TEXT: {text}'
    '''

    prompt = PromptTemplate(
        input_variables=['text', 'language'],
        template=template
    )

    llm.get_num_tokens(prompt.format(text=text, language='english')) # get number of tokens

    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.invoke({'text': text, 'language':'chinese'})
    print(summary)



#Summarization using StuffDocumentChain
if step==3:

    from langchain_core.prompts import PromptTemplate
#    from langchain.chains import LLMChain
    from langchain.chains.summarize import load_summarize_chain
    from langchain.docstore.document import Document

#    with open('files/steve-jobs-speech.txt', encoding='utf-8') as f:
    with open(text) as f:
            text = f.read()

    docs = [Document(page_content=text)]
    template = '''Write a summary of the following text. TEXT: '{text}'
    '''

    prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )

    chain = load_summarize_chain(
        llm,
        chain_type='stuff',
        prompt=prompt,
        verbose=False
    )

    output_summary = chain.invoke(docs)
    print(output_summary)


#Summarising large docs using map reduce
# requires multiple calls to OpenAI and $$$
if step==4:
    from langchain_core.prompts import PromptTemplate
    #    from langchain.chains import LLMChain
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open(text) as f:
            text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50) #paly around settings
    chunks = text_splitter.create_documents(([text]))

    len(chunks) # check the num of chunks - keep it low to prevent too many calls"

    chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        verbose=False
    )
    output_summary = chain.invoke(docs)
    print(output_summary)


#commands to display  what prompts are used
#chain.llm_chain.prompt.template
#chain.combine_document_chain.llm_chain.prompt.template


#Summarising large docs using map reduce & custom prompts
# requires multiple calls to OpenAI and $$$
if step==5:

    map_prompt = '''
    Write a summary of the following:
    Text: {text}
    CONCISE SUMMARY:
    '''

    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )

    combine_prompt = f'''
    Write a summary of all the following text that covers the key pooiints
    Add a title 
    Start summary with an INTRODUCTION that gives an overvew of the topic 
    Followed by BULLET POINTS if possible AND 
    End with a CONCLUSION PHRASE
    Text: {text}
    '''

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm =llm,
        chain_type='map_reduce' ,
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    output = summary_chain.invoke(chunks)  #assumes chunking was done previously
    print(output)



#Summarise using refine CombineDocumentChain
#1. summarize(chunk 1) ==> summary 1
#2 summarize(summary 1 + chunk 2 ==> summary 2
#3 summarize(summary 2 + chunk 3 ==> summary 3
#n summarize(summary n-1 + chunk n ==> summary final summary
# methods can pull in more content
# requires many call to llm

if step == 6:

#    pip install Unstructured
#    pip install pdf2image

    from langchain_core.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import UnstructuredPDFLoader


    loader = UnstructuredPDFLoader('files/xxxx.pdf')
    data= loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)  # paly around settings
    chunks = text_splitter.split_documents(data)

    len(chunks)  # check the num of chunks - keep it low to prevent too many calls"

    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        verbose=False
    )
    output_summary = chain.invoke(chunks)
    print(output_summary)




#Summarise using refine CombineDocumentChain with custom prompts
if step == 7:

#    pip install Unstructured
#    pip install pdf2image

    from langchain_core.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import UnstructuredPDFLoader

    prompt_template = '''
    Write a summary of the following:
    Text: {text}
    CONCISE SUMMARY:
    '''
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    refine_template = '''
    Your job is to produce a final summary.
    I provide an exiting summary up to a certain point: {existing_sanswer}.
    Please refine the existing summary with some more contet below
    -----
    {text}
    ------
    Start final summary with INTRODUCTION PARAGRAPH, that give an overview of the topic
    followed by BULLET POINTS if possible
    AND end the summary with a conclusion phrase
    '''
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_answer', 'text']
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        verbose=False
    )
    output_summary = chain.invoke(chunks)
    print(output_summary)



#Summarise using langchain agent
if step == 8:
    from langchain.agents import initialize_agent, Tool
#    from langchain_community.tools import WikipediaAPIWrapper
    from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
    wikipedia = WikipediaAPIWrapper()

    tools = [
        #define/create tools the agent can use to interact with the outside
        Tool(
            name='Wikipedia',
            func=wikipedia.run,
            description='Used for pulling out wikipedia content'
        )
    ]

    agent_executor = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)
    output= agent_executor.invoke('Can you please provide a summary of George Washington?')
    print(output)