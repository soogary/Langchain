#PROJECT: Custom ChatGPT app

#example 1 no memory or context
#Load your OpenAI API key from .env file
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)



#from langchain.chat_models import ChatOpenAI *depracated*
#from langchain_community.chat_models import ChatOpenAI *depracated*
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

step = 3

if step==1:

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

    prompt = ChatPromptTemplate(
        input_variables=['content'],
        messages=[
            SystemMessage(content='You are a chatbot aving a cnversation with a human.'),
            HumanMessagePromptTemplate.from_template('{content}')
        ]
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    while True:
        content = input('Your prompt: ')
        if content in ['quit', 'exit', 'bye']:
            print('Goodbye')
            break
    
        response = chain.run({'content': content})
        print(response)
        print('-' * 50)
    

    #Use prompts such as "Paris is...", "Its population is ...", "tell me about its history" 
    # - chatbot will not know context for follow up questions
    #change the SystemMessage(content='Respond only in spanish') and check the response
    #change verbose=False and check the response

# Example 2 adding memory buffer

if step==2:
    # part I
    from langchain.memory import ConversationBufferMemory  #ADD memory buffer class 
    from langchain.prompts import MessagesPlaceholder

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)


    # part II memory objects
    memory = ConversationBufferMemory(            #ADD memory object
        memory_key='chat_history',
        return_messages=True
    )

    # part III
    prompt = ChatPromptTemplate(
        input_variables=['content'],
        messages=[
            SystemMessage(content='You are a chatbot aving a cnversation with a human.'),
            MessagesPlaceholder(variable_name='chat_history'), #ADD define where  memory is stored
            HumanMessagePromptTemplate.from_template('{content}')
        ]
    )

    #part IV
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,   #ADD update memory variable argument from above
        verbose=True
    )

    while True:
        content = input('Your prompt: ')
        if content in ['quit', 'exit', 'bye']:
            print('Goodbye')
            break
    
        response = chain.run({'content': content})
        print(response)
        print('-' * 50)

        # test to check memory buffer Q1 "earths mass is ..., Q2 "its diameter is ... Q3 "its distance from the Sun is ... 
        # Q4. " What questions did i ask?"


# Saving chat sessions to a JSON file (or database is also possible)
if step==3:

    # part I import classes
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import MessagesPlaceholder
    from langchain.memory import FileChatMessageHistory  #ADD chat message history class 

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)


    # part II memory & history objects
    history = FileChatMessageHistory('chat_history.json') #ADD history object
    memory = ConversationBufferMemory(            
        memory_key='chat_history',
        chat_memory=history, #ADD keyword argument to convo buffer history constructor
        return_messages=True
    )

    # part III
    prompt = ChatPromptTemplate(
        input_variables=['content'],
        messages=[
            SystemMessage(content='You are a chatbot aving a cnversation with a human.'),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template('{content}')
        ]
    )

    #part IV
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    while True:
        content = input('Your prompt: ')
        if content in ['quit', 'exit', 'bye']:
            print('Goodbye')
            break
    
        response = chain.run({'content': content})
        print(response)
        print('-' * 50)

    #test Q1 Whats the lights speed in a vacuum? Q whats the light speed in water 
    # check the repo for chat_history.json

