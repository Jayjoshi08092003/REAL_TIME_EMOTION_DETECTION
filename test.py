# test.py

import torch
from torchvision import transforms
from PIL import Image
from emotion_model_cnn.model import MiniXception

# Load model
model = MiniXception(num_classes=6)
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load sample image
image_path = "test/smile.jpg"  # Change path
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, 1).item()
    print(f"Predicted class index: {pred_class}")
