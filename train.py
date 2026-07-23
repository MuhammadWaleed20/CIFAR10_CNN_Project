import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import CIFAR_10

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
NUM_CLASSES = 10
NUM_WORKERS = 2

MODEL_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted_classes = outputs.max(dim=1)

        total_images += labels.size(0)

        correct_predictions += (
            predicted_classes == labels
        ).sum().item()

    average_loss = running_loss / len(train_loader)

    accuracy = (
        100.0 * correct_predictions / total_images
    )

    return average_loss, accuracy


def validate(
    model,
    test_loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted_classes = outputs.max(dim=1)

            total_images += labels.size(0)

            correct_predictions += (
                predicted_classes == labels
            ).sum().item()

    average_loss = running_loss / len(test_loader)

    accuracy = (
        100.0 * correct_predictions / total_images
    )

    return average_loss, accuracy


def main():
    print("CIFAR-10 CNN Training")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    print("\nLoading CIFAR-10 dataset...")

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )

    model = CIFAR_10(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    best_validation_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Accuracy: {train_accuracy:.2f}% "
            f"| Validation Loss: {validation_loss:.4f} "
            f"| Validation Accuracy: {validation_accuracy:.2f}% "
            f"| Learning Rate: {current_learning_rate:.6f}"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"Best model saved: {MODEL_PATH} "
                f"({best_validation_accuracy:.2f}%)"
            )

        scheduler.step()

    print("\n" + "=" * 60)
    print("Training completed")
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Best model location: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()