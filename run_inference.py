import os
from inference import run_live_emotion_detection, load_model
import torch

def get_class_names(data_dir):
    train_path = os.path.join(data_dir, 'train')
    return sorted([d.name for d in os.scandir(train_path) if d.is_dir()])

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'my_data')
    model_path = os.path.join(current_dir, 'trained_model.pth')

    # ✅ Load class names dynamically based on folder names
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Ensure num_classes matches your trained model's output
    model = load_model(model_path, device, num_classes=len(class_names))
    run_live_emotion_detection(model, device, class_names)

if __name__ == "__main__":
    main()
