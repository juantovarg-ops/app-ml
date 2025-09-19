import streamlit as st
import requests
from PIL import Image
import pandas as pd
from utils import load_pickle, make_prediction, process_label, process_json_csv, output_batch, return_columns
import os
import pickle
from functools import lru_cache


@lru_cache(maxsize=100, )
def load_pickle(filename):
    with open(filename, 'rb') as file: # read file
        contents = pickle.load(file) # load contents of file
    return contents

cur_dir = os.getcwd()

image_path = os.path.join(cur_dir, 'static/image.jpg')
image = Image.open(image_path)

model_path = os.path.join(cur_dir,'..', 'components', 'model-1.pkl')
transformer_path = os.path.join(cur_dir,'..', 'components', 'preprocessor.pkl')

model = load_pickle(model_path)
transformer = load_pickle(transformer_path)

def predict(pg: float, bwr1: float, bp : float, bwr2: float, bwr3: float, bmi: float, bwr4: float, age: int, insurance: bool):

    data = pd.DataFrame([[pg,bwr1,bp,
                           bwr2,bwr3,bmi, 
                           bwr4, age,insurance]], columns=return_columns())
    
    labels, prob = make_prediction(data, transformer, model)

    response = output_batch(data, labels)
    response_text =  response.json()
    sepsis_status = response_text['results'][0]['0']['output']['Predicted Label']
    return sepsis_status


# set page configuration
st.set_page_config(
    page_title='Sepsis Prediction',
    page_icon="🤖",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a Health App. Call it the Covid Vaccine Sepsis Analyzer!"
    }
)  


# create a sidebar and contents
st.sidebar.markdown("""
## Demo App

This app return sepsis status base on the input parameters
""")

st.markdown('''
    <h1 style="color: green; text-align:center">The Sepsis Prediction App</h1>
    ''', unsafe_allow_html=True)

# insert an image
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


# Create app interface
container = st.container()
container.write("**Inputs to predict Sepsis**")
with container:
    col1, col2, col3 = st.columns(3)
    
    age = col1.number_input(label='Age')
    pg = col2.number_input(label='Blood Glucose')
    bp = col3.number_input(label='Blood Pressure')
    with st.expander(label='Blood Parameter', expanded=True, ):
        bwr1 = col1.number_input(label='Blood Work Result-1')
        bwr2 = col2.number_input(label='Blood Work Result-2')
        bwr3 = col1.number_input(label='Blood Work Result-3')
        bwr4 = col2.number_input(label='Blood Work Result-4')
    ins = col3.selectbox(label='Insurance', options=[True, False])
    bmi = col3.number_input(label='Body Mass Index')
    button = st.button(label='Predict', type='primary', use_container_width=True)
    

if button:
    response = predict(pg, bwr1, bp, bwr2, bwr3, bmi, bwr4, age, ins)
    st.write(response)
