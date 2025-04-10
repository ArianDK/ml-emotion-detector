import os
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate_model

def train_model(model, train_loader, val_loader, class_names, num_epochs=10, learning_rate=0.001, device='cpu', save_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate_model(model, val_loader, device, class_names)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path is None:
                project_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(project_dir, "trained_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at: {save_path}")

    print("🎉 Training complete.")
