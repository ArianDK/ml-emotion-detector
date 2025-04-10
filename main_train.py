import os
import torch
from data_loader import get_data_loaders
from model import build_model
from train import train_model
from inference import get_class_names

def main():
    print("Starting training...")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset")
    save_path = os.path.join(base_dir, "trained_model.pth")

    # Dynamically load class names from folder names
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    print(f"✅ Loaded data. Classes: {class_names}")

    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(data_dir=data_dir, grayscale=True)

    # Build model
    model = build_model(num_classes=num_classes)
    print("✅ Model built.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Train
    train_model(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        num_epochs=10,
        learning_rate=0.001,
        device=device,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
