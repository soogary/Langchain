# Building a custom CHATGPT app

import os
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv() , override=True)

from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import streamlit as st

from streamlit_chat import message #this function presents chats in a whatsapp style format

# add web page title & icon if any
st.set_page_config(
    page_title='Your Custom Assistant'
    #page_icon=''
)
st.subheader('your custom ChatGPT 🤙')
chat  = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

#save entire chat history in streamlit session state: key = 'messages'
if 'messages' not in st.session_state:
    st.session_state.messages = [] #initializing with empty list


#use sidebar to create input fields and behaviors
with st.sidebar:
    system_message = st.text_input(label='System role') #widget 1
    user_prompt = st.text_input(label='Send a message') #widget 2
    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages): #i.e. if no sys msgs exist
            st.session_state.messages.append(
                SystemMessage(content=system_message)
            )

    #st.write(st.session_state.messages)  #debugging: show session state msgs eg if any SystemMessage exists

    #handle user request
    if user_prompt:
        st.session_state.messages.append(
            HumanMessage(content=user_prompt)
        )

        with st.spinner('working on your request ...'):
             response = chat(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=response.content))

        #st.write(st.session_state.messages)  #debugging: show session state msgs eg if any SystemMessage exists

#st.session_state.messages

# design the chat output style
#message('this is chatgpt', is_user=False)
#message( 'this is the user', is_user=True)

if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, SystemMessage(content='you are a helpful assistant.'))

for i, msg in enumerate(st.session_state.messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=f'{i}+😎')
    else:
        message(msg.content, is_user=False, key=f'{i}+🧟')

