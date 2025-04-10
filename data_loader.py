import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, image_size=224, grayscale=False):
    # Define normalization values
    if grayscale:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Extract class names
    class_names = train_dataset.classes

    return train_loader, val_loader, class_names
