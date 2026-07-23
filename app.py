import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from src.model import CIFAR_10

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


@st.cache_resource
def load_model():
    model = CIFAR_10(num_classes=10)

    model.load_state_dict(
        torch.load(
            "models/best_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = classes[predicted.item()]
    confidence = confidence.item() * 100

    return predicted_class, confidence


st.title("CIFAR-10 Image Classifier")

st.write(
    "Upload an image and the CNN model will classify it into one "
    "of the ten CIFAR-10 categories."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        predicted_class, confidence = predict_image(image)

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )