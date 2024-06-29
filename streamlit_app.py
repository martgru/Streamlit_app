#streamlit_app.py

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
import torch


model_path = 'model/'
tokenizer_path = 'model/tokenizer/'
generation_config_path = 'model/generation_config.json'

model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
generation_config = GenerationConfig.from_pretrained(generation_config_path)


st.title('Japanese to English Translator')
st.write('Enter a sentence.')

japanese_text = st.text_area('Japanese Text', '')

if st.button('Translate'):
    if japanese_text:
        inputs = tokenizer(japanese_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        translated_tokens = model.generate(**inputs, generation_config=generation_config)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        st.write('Translated Text:', translated_text)
    else:
        st.write('Please enter a sentence.')
