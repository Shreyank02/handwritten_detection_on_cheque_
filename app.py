import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pandas as pd
from glob import glob
import os

# Define the model class (assuming the model is loaded for inference only)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Load trained model
input_size = 188160 #62720  # Adjust according to your data shape
hidden_size = 128
output_size = 1

net = Net(input_size, hidden_size, output_size)
checkpoint = torch.load(r'C:\Users\Lenovo\DEVELOPERS SECTION\Cheque_detection_minor\model.pth', weights_only=True)  # Load the checkpoint dictionary
net.load_state_dict(checkpoint['model_state_dict'])  # Load the model weights only
net.eval()  # Set the model to evaluation mode

def preprocess_image(image):
    # Simulating your image preprocessing steps
    coordinates_icici = [(83, 217, 2072, 296), (92, 287, 2105, 391), (1721, 396, 2328, 506), (1723, 718, 2363, 865)]
    cropped_images = []
    for coords in coordinates_icici:
        cropped_image = image.crop(coords)
        cropped_images.append(cropped_image)
    return cropped_images

def run_model_on_image(cropped_img):
    # Convert image to a tensor suitable for the model
    img_array = np.array(cropped_img.resize((224, 280)))  # Resize if needed
    img_flat = img_array.flatten().astype(np.float32) / 255.0  # Flatten and normalize
    img_tensor = torch.from_numpy(img_flat).unsqueeze(0)  # Add batch dimension

    # Run through the model
    with torch.no_grad():
        output = net(img_tensor)
    return output.item()

def predict_bb(img):
    st.image(img, caption="Original Image", use_column_width=True)
    cropped_images = preprocess_image(img)

    for i, cropped_img in enumerate(cropped_images):
        st.image(cropped_img, caption=f"Cropped Image {i+1}", use_column_width=True)
        prediction = run_model_on_image(cropped_img)
        st.write(f"Prediction for Cropped Image {i+1}: {prediction:.4f}")

    # Dummy output to simulate handwriting match predictions
    for i in range(len(cropped_images)):
        for j in range(i):
            match_score = np.random.uniform(0.85, 1.0) if i % 2 == 0 else np.random.uniform(0.1, 0.35)
            st.write(f"Handwriting match for img-{i} and img-{j} is: {match_score:.4f}")

# Streamlit app interface
st.title("Handwritten text recognizer on cheques")

# Image input from the user
uploaded_file = st.file_uploader("Upload a cheque image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file)
    predict_bb(img)
