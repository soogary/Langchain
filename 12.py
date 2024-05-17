# Building with STREAMLIT

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

import pandas as pd


#write command

st.header('header')

st.subheader('subheader')

st.write('a line of text')

#magic
'show this text'

df = list[1,2,3,4]
df

#text imputs
name = st.text_input('enter name =')
if name:
    st.write(f'Hello {name}')

#numeric inputs
x = st.number_input('enter number', min_value=1, max_value=99, step=1)
st.write(f'current number is {x}')

#divider
st.divider()

#button
clicked = st.button('click here') #returns T or F
if clicked:  # is True
    st.write(':ghost:' * 3)

#divider
st.divider()

#checkbox
agree = st.checkbox('i agree') #returns T or F
if agree:
    'Great you agree'

checked = st.checkbox('continue', value=True)
if checked:
    'thats Great too '


#display table triggerd by checkbox
table=pd.DataFrame({'Name':['Mario', 'Luigi'],
                 'Age':[12,11]
                 })

if st.checkbox('show me data'):
    st.write(table)

#radio button
pets = ['dog','cat','fish','hamster']
pet = st.radio('favorite pet', pets, index=2)
st.write(f'your fave pet is {pet}')


#pull down selection
cities = ['London', 'Berlin', 'Paris', 'Madrid']
city = st.selectbox('your city', cities, index=1)
st.write(f'you live in {city}')

#Slider
x = st.slider('x')
st.write(f'x is{x}')


y = st.slider('x', value=15, min_value=10, max_value=90, step=5)
st.write(f'y is{y}')


#File uploader
upload_file_1 = st.file_uploader('Uploaded a file:')

upload_file_2 = st.file_uploader('Uploaded a file:', type=['txt','csv','xlsm'])

if upload_file_2:
    st.write(upload_file_2)
    if upload_file_2.type == 'text/plain':
        from io import StringIO
        stringio = StringIO(upload_file_2.getvalue().decode('utf-8'))
        string_data = stringio.read()
        st.write(string_data)
    elif upload_file_2.type == 'text/csv':
        import pandas as pf
        df = pd.read_csv(upload_file_2)
        st.write(df)
    else:
        import pandas as pd
        df = pd.read_excel(upload_file_2)
        st.write(df)

#camera input - capture and display image of user webcan
#camera_photo = st.camera_input('take photo')
#if camera_photo:
#    st.image(camera_photo)


#camera_photo = st.image('https://a-url')


#sidebar layout
my_select_box = st.sidebar.selectbox('select', ['US', 'UK', 'DE','FR'])
my_slider = st.sidebar.slider('temperature')

#displaying with column layouts
l_col, r_col = st.columns(2)

import random
data= [random.random() for _ in range(100)]

with l_col:
    st.subheader('a linechart')
    st.line_chart(data)

r_col.subheader('Data')
r_col.write(data[:10])

col1, col2, col3 = st.columns([0.2, 0.3, 0.5])  #sets % width per column"

col1.markdown("hello world")
col2.write(data[5:10])

with col3:
    st.header('A cat')
    st.image('https://static.streamlit.io/examples/cat.jpg')


#expander container
with st.expander('click to expand'):
    st.bar_chart({'Data':[random.randint(2,10) for _ in range(25)]})
    st.write('this is an image of a dog')
    st.image('https://static.streamlit.io/examples/dog.jpg')


#progress bar
import time

st.write('starting a long computation')
latest_iteration = st.empty() # contain in app to hold some elements

progress_text = 'operation in progress ...'
my_bar = st.progress(0, text=progress_text)
time.sleep(2)

for i in range(2):
    my_bar.progress(i+1)
    latest_iteration.text(f'Iteration {i+1}')
    time.sleep(0.1)

st.write('we are done')


# working with session states
# a session is a python object that exists in memory for use between runs
# a session state is a way to share variables between runs
# sessions on different browser tabs are independent

st.title('streamlit session')
st.write(st.session_state) #session state is a dict. This is an empty session state

if 'counter' not in st.session_state: #create a counter that persists across  different page refreshes
    st.session_state['counter'] = 0
else:
    st.session_state.counter += 1

st.write(f'Counter: {st.session_state.counter}')   #hit Rerun or type 'r' in the browser
#st.write(st.session_state.counter)

button = st.button("update state")
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = 0

if button:
    st.session_state['clicks'] += 1
    f'after pressing button {st.session_state}'

#connect a widget to session state
number = st.slider('Value', 1, 10, key='my_slider')
st.write(st.session_state)
st.write(number)

#callbacks
#function called when user interacts with widget On_change  or 0n_click

st.subheader("distance converter")

def miles_to_km():
    st.session_state.km = st.session_state.miles * 1.609

def km_to_miles():
    st.session_state.miles = st.session_state.km * 0.621

col_a, buff, col_b = st.columns([2,1,2])
with col_a:
    miles = st.number_input("miles:", key='miles', on_change=miles_to_km)

with col_b:
    km = st.number_input('km:', key='km', on_change=km_to_miles)




