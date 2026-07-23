import torch
from torchvision import transforms
from PIL import Image

from src.model import CIFAR_10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {device}")

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

model = CIFAR_10(num_classes=10)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded Successfully")


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = classes[predicted.item()]

    confidence = confidence.item() * 100

    return predicted_class, confidence


def main():

    image_path = input("Enter image path: ")

    predicted_class, confidence = predict_image(image_path)

    print("\nPrediction Result")
    print("----------------------------")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%")
    print("----------------------------")


if __name__ == "__main__":
    main()