import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import build_model
from torchvision import transforms

def get_class_names(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return class_names

def load_model(weights_path, device, num_classes, in_channels=1):
    model = build_model(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_live_emotion_detection(model, device, class_names):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)
    print("🎥 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)  # For visualization
            face_tensor = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)

            emotion = class_names[predicted.item()]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Live Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
