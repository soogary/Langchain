import os
import langchain
import langchain_openai
#Load API keys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

os.environ.get('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI

step = 10

if step==1:
#Using the openAI GPT chat model
    llm = ChatOpenAI()
    #output = llm.invoke('Explain quantum mechanics in 1 sentence')
    output = llm.invoke('Explain quantum mechanics in 1 sentence', model='gpt-3.5-turbo')
#    output = llm.invoke('Tell me a joke that a toddler can understand', model='gpt-3.5-turbo')
    print(output.content)


if step==2:
    #help - check the model being used
    help(ChatOpenAI)


#in memory caching
if step==3:
    from langchain.globals import set_llm_cache
    from langchain_openai import OpenAI
    llm = OpenAI(model_name='gpt-3.5-turbo')
    #print(output.content)

# to measure response time of model
#    %%time
    from langchain.cache import InMemoryCache
    set_llm_cache(InMemoryCache())
    prompt = 'Tell me a joke that a toddler can understand.'

#will shw response time stats uncached first run

#to request again from cache
    llm.invoke(prompt)


#STREAMING
#non Streaming response

if step==4:
#non stream example 

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
    prompt = 'write a rock song about the moon and a raven'
    print(llm.invoke(prompt).content)


if step==5:
#streaming example - observe the behavior of stream
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
    prompt = 'write a rock song about the moon and a raven'

    llm.stream(prompt)
    for chunk in llm.stream(prompt):
        print(chunk.content, end='', flush=True) 


#Prompt Templates / 
if step==6:
    from langchain.prompts import PromptTemplate
    template = '''your are an expenienced virologist. write a few sentences about the following {virus} in {language}'''
    prompt_template = PromptTemplate.from_template(template = template)

    prompt = prompt_template.format(virus='hiv', language='chinese')
    prompt

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    output = llm.invoke(prompt)
    print(output.content)


#Chat Prompt Templates
if step==7:
#create a output following system output in json format

    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain_core.messages import SystemMessage

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content='You respond only in the JSON format'),
            HumanMessagePromptTemplate.from_template('Top {n} countries in {area} by population.')
        ]
    )

    messages = chat_template.format_messages(n='10', area='Europe')
    print(messages)

    llm = ChatOpenAI()
    output = llm.invoke(messages)
    print(output.content) 


if step==8:
#Simple chains using prompt template with 2 variables as dictionary
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain

    llm = ChatOpenAI()
    template = '''your are an experienced virologist. write a few sentences about the following {virus} in {language}'''
    prompt_template = PromptTemplate.from_template(template = template)

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
#        verbose=True
    )

    output = chain.invoke({'virus': 'HSV', 'language': 'Spanish'})
    print(output)

if step==9:
#Simple chains using template with 1 variable as string
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain

    llm = ChatOpenAI()
    template = 'What is the capital of {country}?. List the top 3 places to visit in that city. Use bullet points'
    prompt_template = PromptTemplate.from_template(template = template)

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    #    verbose=True
    )

    country = input('Enter Country: ')
    output = chain.invoke(country)
    print(output['text'])



if step==10:
#Sequential Chains (simple)
# the output of an LLM triggered from chain 1 becomes an input for an LLM called in chain 2 

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain
                                        #import sequential chain class

    llm1 = ChatOpenAI(model_name ='gpt-3.5-turbo', temperature=1.2)
    prompt_template1 = PromptTemplate.from_template(
        template='You are an experienced scientiest and Python programmer. Write a function that implements the concept of {concept}.'
    )
    chain1 = LLMChain(llm=llm1, prompt=prompt_template1)


    llm2 = ChatOpenAI(model_name ='gpt-3.5-turbo', temperature=0.5)
    prompt_template2 = PromptTemplate.from_template(
            template='Given the Python function {function} describe it in as much detail as possible.'
    )
    chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
    output = overall_chain.invoke('linear regression')

    print(output['output'])
