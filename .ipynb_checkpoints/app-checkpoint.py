import streamlit as st
import torch
import torch.nn as nn

st.title('Handwritten text detection on cheques')

uploaded_file = st.file_uploader("Upload an Image", type="jpg")

if uploaded_file is not None:
    model = torch.load(r'DEVELOPERS SECTION/Cheque_detection_minor/model.pth')
    image = Image.open(uploaded_file)
    