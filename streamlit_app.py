import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from lenet_5.model import LeNet
from streamlit_mnist_canvas import st_mnist_canvas

footer = """<style>
    a:link, 
    a:visited{ 
        color: blue; 
        background-color: transparent; 
        text-decoration: underline; 
    } 

    a:hover, 
    a:active { 
        color: red; 
        background-color: transparent; 
        text-decoration: underline; 
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        z-index: 9999; /* Ensure footer stays on top */
        display: block;
    }
</style>

<div class="footer">
    <p>
        Developed by 
        <a style="color: black;" href="https://chloelavrat.com" target="_blank">
            <b>Chloé Lavrat</b>
        </a>
    </p>
</div>"""

# Set up page
st.set_page_config(
    page_title="LeNet-5",
    page_icon="🧠",
    layout="centered")

st.markdown(
    "<style>#MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)
st.markdown(footer, unsafe_allow_html=True)

# Load the saved model
model = LeNet()
model.load_state_dict(torch.load(
    'saved_models/lenet_mnist.pth')['model_state_dict'])
model.eval()

# Define transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])


def predict_digit(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item(), output


# Streamlit app layout
st.image('./assets/banner.png')
st.markdown("""**LeNet-5** is a convolutional neural network developed by Yann LeCun et al. in 1998 specifically for recognizing handwritten digits. The architecture includes two convolutional layers, each followed by average pooling, and concludes with fully connected layers. This model is extensively utilized for various image classification tasks.
## How to Use
- Draw a digit in the canvas.
- Click the submit button to get the prediction.
- View the predicted digit and probability distribution.
## Model Performance
The model has been trained on the MNIST dataset and achieves a high accuracy in recognizing handwritten digits. You can try different digits and see how well it performs!
## Demo
Draw a digit in the canvas: 
""")

# Create a canvas for user input
result = st_mnist_canvas()

if result.is_submitted:
    col1, col2 = st.columns([1, 1])
    col1.subheader("LeNet-5 Output:")

    try:
        prediction, output = predict_digit(
            Image.fromarray(result.resized_grayscale_array))
        col2.success(f"The predicted digit is: {prediction}")

        with st.expander("Probability Distribution"):
            st.bar_chart(
                output[0, :], x_label="Detected Value", y_label="Probability")

    except Exception as e:
        col2.error(f"Error occurred: {str(e)}")
