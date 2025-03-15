import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Set page config
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")

# Welcome message
st.title("✏️ Handwritten Digit Recognizer")
st.markdown("""
Welcome to the **Handwritten Digit Recognizer**! Upload an image of a handwritten digit (0-9), and this app will predict what number it is using a lightweight AI model.
""")

# Define lightweight CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 10)  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    try:
        state_dict = torch.load("mnist_simplecnn.pth", map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model file 'mnist_simplecnn.pth' not found! Please train it using train_mnist.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    model.eval()
    return model

model = load_model()

# Image preproces
transform = transforms.Compose([
    transforms.Grayscale(),          
    transforms.Resize((28, 28)),     
    transforms.ToTensor(),           
    transforms.Normalize((0.1307,), (0.3081,)) 
])

st.subheader("Upload Your Digit")
st.markdown("*Upload a grayscale image of a handwritten digit (0-9).*")
uploaded_file = st.file_uploader("Choose an image....!", type=["jpg", "png", "jpeg"])

st.markdown("*Click to predict the digit.*")
predict_button = st.button("Recognize Digit")

# Process and display output
if predict_button and uploaded_file is not None and model is not None:
    with st.spinner("Recognizing..."):

        image = Image.open(uploaded_file)
        image_tensor = transform(image).unsqueeze(0) 
        
        # prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_digit = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_digit].item()

        # Display result
        st.image(image, caption="Uploaded Digit", width=150)
        st.success(f"Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.2%})")
        
elif predict_button and uploaded_file is None:
    st.warning("Please upload an image first!")
elif predict_button and model is None:
    st.error("Model not loaded.")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stFileUploader {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stSpinner {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)