import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, data_loader, device, class_names):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        class_names (list): List of class names.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return accuracy
