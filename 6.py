#Load API keys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
#Google Gemini and Langchain
#get your API Key from GoogleAI

pip install langchain-google-genai
pip install langchain-google-generativeai

pip show langchain-google-genai

import getpass
import os
if 'google_API_KEY' not in os.environ
    os.environ['GOOGLE_API_KEY'] = get.pass('Provide your Google API key')

import google.generativeai as genai
for model in genai.list_models():
    print(model.name)


from langchain_google_genai import ChatGoogleGnerativeAI
#llm = ChatGoogleGnerativeAI(model='gemini-pro', temperature=0.9, google_api_key-'xxxxxxxx')
llm = ChatGoogleGnerativeAI(model='gemini-pro', temperature=0.9)
response = llm.invoke('Write a paragraph about life on Mars in year 2100')
print(response.content)


#use prompt tempalte
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
llm = ChatGoogleGnerativeAI(model='gemini-pro')

prompt = PromptTemplate.from_template('your are a content creator. Write me a tweet about {topic}')
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

topic = 'Why will AI change the world'

response = chain.invoke(input=topic)
print(response)

print(response['topic'])
print()
print(response['text'])


#System prompt and streaming
from langchain_core.message import HumanMessage, SystemMessage
llm = ChatGoogleGnerativeAI(model='gemini-pro', convert_system_message_to_human=True)
output = llm.invoke(
    [
        SystemMessage(content='Answer only YES or No in French.'),
        HumanMessage(content='Is cat a mammal?')
    ]
)
output.content


#Streaming
llm = ChatGoogleGnerativeAI(model='gemini_pro', temperature=0)
prompt = 'Write a scientic paper outlining te mathematical foundation of our universe.'
response= llm.invoke(prompt)
print(response.content)

for chunk in llm.stream(prompt)
    print(chunk.content, end='')
    print('-' * 100)

#multimodel AI with Gemini Pro Vision
pip install pillow
from PIL import Image
img = Image.open('match.jpg')

from langchain_core.message import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-pro-vision')
prompt = 'what is in this image?'
message = HumanMessage(
    content[
        {'type': 'text', 'text':prompt},
        {'type': 'image_url', 'image_url': img}
    ]
)

response = llm.invoke([message])
print(response.content)

#
def ask_gemini(text, image, model='gemini-pro-vision')
    llm = ChatGoogleGenerativeAI(model=model)
    message = HumanMessage(
        content[
            {'type': 'text', 'text':prompt},
            {'type': 'image_url', 'image_url': img}
        ]
    )
    response = llm.invoke([message])
    return response

response = ask_gemini('What is the sport" How can i identify the soport in this ')
print(response.content)

response = ask_gemini('How many players are in each team')
print(response.content)

#

import requests
from Ipython.display import Image
image_url = 'http://picsum etc xxx '
content = requests.get(image_url).content
image_data = Image(content)
image_data

#API call to Gemini
reponse = ask_gemini('describe this image as detailed as possible', image_url)
print(response)

#gemini Safety settings
from langchain_google_genai import ChatGoogleGenerativeAI
llm1 = ChatGoogleGenerativeAI(model='gemini_pro')

prompt = 'How to shoot an animal?'
reponse = llm1.invoke(prompt)
print(response.content)
# expect the request to be blocked

from langchain_google_genai import HarmCategory, HarmBlockThreshold
llm2 = ChatGoogleGenerativeAI(
    model='gemini-pro',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARRASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
)

response = llm2.invoke(prompt)
print(response.content)
#view the behavior / message


