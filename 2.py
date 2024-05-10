import os
import langchain
import langchain_openai
#Load API keys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

os.environ.get('OPENAI_API_KEY')


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
output = llm.invoke('Explain quantum mechanics in 1 sentence')
#output = llm.invoke('Explain quantum mechanics in 1 sentence', model='gpt-4-turbo-preview')
print(output.content)

#help - check the model being used
help(ChatOpenAI)

#in memory caching
from landchain.globals import set_llm_cach
form langchain_openai import OpenAI
llm = OpenAI(model_name='gpt-4-turbo-instruct')

# to measur response time of model
%%time
from langchain.cache import InMemoryCacheset_llm_cache(inMemoryCache())
prompt = 'Tell me a joke that a toddler can understand.'

#will shw response time stats uncached first run

#to request again from cache
llm.invoke(prompt)

#STREAMING
#non Streaming response
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
prompt = 'write a rock song about the moon and a raven'
print(llm.invoke(prompt).content)


#streaming example
llm.stream(prompt)


#to observe the behavior of stream
for chunk in llm.stream(prompt):
    print(chunk.content, end='', flush=True) 


#Prompt Templates / 
from langchain.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
template = '''your are an expenienced virologist. write a few sentences about the following {virus} in {language}'''
prompt_template = PromptTemplate.from_template(template = template)

prompt = prompt_template.format(cirus='hiv', language='german')
prompt

llm = ChatOpenAI(model_name='gpt-3.5.turbo', tempaerature=0)
output = llm.invoke(prompt)
print(output.content)


#Chat Prompt Templates
#create a output following system output in json format

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_message(

    SystemMessage(content='You reposnd only in the JSON format'),
    HumanMessagePromptTemplate.from_template('Top {n} countries in {area} by population')

)

messages = chat_template.format_messages(n='10', area='Europe')
print(messages)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
output = llm.invoke(messages)
print(output.content) 


#Simple chains using prompt template with 2 variables as dictionary
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm - ChatOpenAI()
template = '''your are an expenienced virologist. write a few sentences about the following {virus} in {language}'''
prompt_template = PromptTemplate.from_template(template = template)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template
#    verbose=True
)

output = chain.invoke({'virus': 'HSV', 'language': 'Spanish'})
print(output)

#Simple chains using template with 1 variable as string
template = 'What is the capital of {country}?. List the top places to visit in that city. Use bullet points'
prompt_template = PromptTemplate.from_template(template = template)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template
#    verbose=True
)

country = input('Enter Country: ')
output = chain.invoke(country)
print(output['text'])


#Sequential Chains (simple)
# chain 2 takes the output of chain 1 as its input

from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm1 = ChatOpenAI(model_name ='gpt-3.5-turbo', temparature=1.2)
prompt_template1 = PromptTemplate.from_template(
        template='You are an experienced scientiest and Python programmer. Write a function that implements the concept of {concept}.'
)
chain1 = LLMChain(llm=llm1, prompt=prompt_template1)


llm2 = ChatOpenAI(model_name ='gpt-4.5-turbo-preview', temparature=0.5)
prompt_template2 = PromptTemplate.from_template(
        template='Given the Pythion function {function} describe it in as much detail as possible.'
)
chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
output = overall_chain.invoke('linear reqgression')

print(output['output'])

