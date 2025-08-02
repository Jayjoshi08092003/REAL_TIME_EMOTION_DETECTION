# predict_realtime.py – Real-Time Emotion Detection with MiniXception
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import create_model
from PIL import Image
import numpy as np

# --- Settings ---
model_path = "best_model_mini_x.pth"
num_classes = 7
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = create_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

# --- Transform ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Face Detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Real-Time Webcam Loop (Try Multiple Backends) ---
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, 0]  # try fallback
cap = None
for backend in backends:
    temp_cap = cv2.VideoCapture(0, backend) if isinstance(backend, int) else cv2.VideoCapture(0, backend)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"✅ Webcam opened using backend: {backend}")
        break
    temp_cap.release()

if not cap or not cap.isOpened():
    print("❌ Failed to open webcam with any backend. Exiting.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
        roi_pil = Image.fromarray(roi)
        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(roi_tensor)
            pred = torch.argmax(output, 1).item()
            label = class_names[pred]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
