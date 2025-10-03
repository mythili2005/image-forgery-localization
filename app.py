import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

# Load model
@st.cache_resource
def load_model():
    model = torch.load("model/forgery_model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

st.title("Image Forgery Detection 🔍")

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Example: output = sigmoid + threshold
        mask = (torch.sigmoid(output) > 0.5).float()

    st.image(mask.squeeze().numpy(), caption="Predicted Forgery Mask", use_column_width=True)
