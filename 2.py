import os
import langchain
import langchain_openai


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


#Prompt Templates / Chat Prompt Templates
from langchain.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
template = '''your are an expenienced virologist. write a few sentences about the following {virus} in {language}'''
prompt_template = PromptTemplate.from_template(template = template)

prompt = prompt_template.format(cirus='hiv', language='german')
prompt

llm = ChatOpenAI(model_name='gpt-3.5.turbo', tempaerature=0)
output = llm.invoke(prompt)
print(output.content)