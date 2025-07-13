import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
data_dir = r'C:\Users\arian\projects\python\20250401_ml_emotion_detector\full_dataset\train'
image_size = (48, 48)
num_images = 32
cols = 8
rows = 4
random_seed = 42

# Set seed for reproducibility
random.seed(random_seed)

# Mislabelled image positions (0-based flattened indices)
mislabelled_positions = [3, 11, 16, 18, 7, 23, 26]     # Red rectangles
questionable_positions = [0, 19, 22, 28]               # Yellow rectangles

# Get all categories (subdirectory names)
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Collect image paths and labels
all_images = []
for category in categories:
    category_path = os.path.join(data_dir, category)
    for img_name in os.listdir(category_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(category_path, img_name)
            all_images.append((img_path, category))

# Randomly sample images
sampled_images = random.sample(all_images, num_images)

# Plotting
fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
axes = axes.flatten()

for idx, (ax, (img_path, label)) in enumerate(zip(axes, sampled_images)):
    try:
        img = Image.open(img_path).convert("RGB").resize(image_size)
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis('off')

        # Fix the axis limits so the rectangle matches the image bounds
        ax.set_xlim([0, image_size[0]])
        ax.set_ylim([image_size[1], 0])  # Invert y-axis to match image orientation

        if idx in mislabelled_positions:
            rect = patches.Rectangle(
                (0, 0),
                image_size[0],
                image_size[1],
                linewidth=4,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

        # Draw yellow rectangle for questionable images
        if idx in questionable_positions:
            yellow_rect = patches.Rectangle(
                (0, 0),
                image_size[0],
                image_size[1],
                linewidth=4,
                edgecolor='yellow',
                facecolor='none'
            )
            ax.add_patch(yellow_rect)

    except Exception as e:
        ax.set_title("Error", fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.show()
