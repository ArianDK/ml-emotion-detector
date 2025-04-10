import cv2
import os
from datetime import datetime

def capture_images(output_dir):
    """
    Captures images from the webcam upon pressing the 'c' key and displays the captured image.
    Each image is saved with a unique filename based on the current timestamp.

    Args:
        output_dir (str): Directory to save captured images.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture an image, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Webcam Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Generate a unique filename using the current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_name = os.path.join(output_dir, f"image_{timestamp}.jpg")

            # Save the captured image
            cv2.imwrite(img_name, frame)
            print(f"Image saved as {img_name}.")

            # Display the captured image
            cv2.imshow('Captured Image', frame)
            cv2.waitKey(500)  # Display the captured image for 500 ms
            cv2.destroyWindow('Captured Image')

        elif key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
capture_images(output_dir='dataset/val/neutral')

# Emotions
# happy
# angry
# sad
# surprise
# neutral