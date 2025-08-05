import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the same model architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("C:\\Federated_learning_project\\FashionMNIST\\federated_fashionmnist.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# FashionMNIST classes
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("FashionMNIST Federated Model Inference")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (PNG/JPG)")

def preprocess(image: Image.Image):
    # Convert to grayscale 28x28
    image = image.convert("L").resize((28, 28))
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)  # batch dimension
    return tensor

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=150)

    input_tensor = preprocess(img)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.write(f"**Prediction:** {classes[predicted.item()]}")
    st.write(f"**Confidence:** {confidence.item()*100:.2f}%")
else:
    st.write("Upload an image to classify it.")

