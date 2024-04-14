import streamlit as st 
import pdb
from streamlit_extras.add_vertical_space import add_vertical_space 
from PIL import Image
import io
from search import get_relevant_objects, append_to_faiss_index
import logging
import os
import torch
import pandas as pd
# IMPORT VLLM
from ml.call_vllm import call_vision_api
import time
import h5py
import numpy as np

from transformers import AutoImageProcessor, AutoModel
import faiss
st.set_page_config(layout="wide")
os.makedirs("input_images", exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
images_prompt = []

#@st.cache_resource
def load_models():

    device ="cpu"
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

#model, processor = load_models()

model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
@st.cache_data
def load_data_objects():
    faiss_index = faiss.read_index("faiss.index")
    test_dataframe = pd.read_csv('test.csv')
    st.session_state.dataframe = test_dataframe
    return faiss_index, test_dataframe

fs_index, dataframe = load_data_objects()

def init_keys():
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False 
    if 'search_started' not in st.session_state:
        st.session_state['search_started'] = False 
    if 'objects_found' not in st.session_state:
        st.session_state['objects_found'] = None
    if 'placeholder' not in st.session_state:
        st.session_state['placeholder'] = "Bведите описание экспоната"

    if 'front_image' not in st.session_state:
        st.session_state['front_image'] = None
    if 'api_calls' not in st.session_state:
        st.session_state['api_calls'] = 0

init_keys()

def update_description():
    pass
def file_uploaded():
    st.session_state['file_uploaded'] = True
    st.session_state['search_started'] = False
    st.session_state['objects_found'] = None

def search_callback():
    st.session_state['search_started'] = True

@st.cache_data
def set_found_objects(objects):
    st.session_state.objects_found = objects

def get_object_info_h5(index: int):
    with h5py.File('data/dataset.h5', 'r') as f:
        object_info = f['objects'][index]

        image_data = object_info['image']
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = Image.open(io.BytesIO(image_data))

        name = object_info['name'].decode()
        desc = object_info['description'].decode()

        return image, name, desc

@st.cache_data
def get_topk_objects(top_k:int):

    return [(get_object_info(i)) for i in range(top_k)]

col1, col2 = st.columns((1, 2))
with col1:
    file_ = st.file_uploader("image_upload", 
                          type=['png', 'jpg', 'jpeg'], 
                          accept_multiple_files=True, 
                          on_change=file_uploaded)
    if file_ is not None and file_:
        logging.warning(file_) 
        latest_file_path = os.path.join("input_images", file_[0].name)
        st.session_state.front_image = latest_file_path
        for uploaded_file in file_:
            bytes_data = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(bytes_data))
            img_save_path = os.path.join("input_images", uploaded_file.name)
            images_prompt.append(img_save_path)
            img.save(img_save_path)
    
    if st.session_state['file_uploaded'] and st.session_state.front_image :
        im = Image.open(st.session_state.front_image)
        st.image(im, width=384)


    if st.button("find similar", 
                 key="find", 
                 disabled=not st.session_state['file_uploaded'],
                 on_click=search_callback
                 ):

        images = [Image.open(path) for path in images_prompt]
        
        objects = get_relevant_objects(
            images, 
            fs_index, 
            dataframe, 
            model=model, 
            processor=processor
        )
        set_found_objects(objects)
        logging.warning(objects)

with col2:
    tile = col2.container(height=500)
    if "objects_found" in st.session_state and st.session_state.objects_found is not None:
        start_show = st.select_slider("choose from relevant objects", options = list(range(0, 10)), value = 5)
        datas = get_topk_objects(start_show)
        for image_ind in range(start_show):
            image,name,  desc = datas[image_ind]


            tile.image(image,caption=name,  width=200)
            tile.text_area(" ", desc, key=f'{image_ind}', height=180)

add_vertical_space(5)
col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    text_input = st.text_area(
            " ",
            disabled=not st.session_state.search_started,
            value=st.session_state.placeholder, 
            on_change=update_description,
            key = "user_description_"
    )

def call_api_callback():
    st.session_state.api_calls +=1
welcome_string = "Помощь в составлении описания :sparkles:"
with col_input2:
    condition = st.session_state.api_calls > 3 and st.session_state.search_started
    if st.button(welcome_string, key='api_call', on_click=call_api_callback, disabled=condition) and st.session_state.front_image :
        image_path = st.session_state.front_image 
        logging.warning(image_path)
        # encoded = encode_image(image_path)
        # CALL VLLM
        descriptions =list(map(str,  st.session_state.objects_found.description.values[:10]))
        try:
            output_text = call_vision_api(descriptions=descriptions, image=Image.open(image_path))
            
        except Exception as e:
            logging.warning(e)
            time.sleep(1)

            output_text = "Что-то пошло не так при генерации описания"

        st.text_area("", output_text, height=200, key="output_generated")
        logging.warning(output_text)
        st.session_state.placeholder = output_text

if st.button("Внести описание в базу", disabled= not st.session_state.search_started):
    logging.warning(st.session_state.user_description)

