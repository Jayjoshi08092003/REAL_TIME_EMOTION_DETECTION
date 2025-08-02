import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import create_model

# --- Configuration ---
data_dir = r'C:\Users\Asus\Desktop\New folder (2)\emotion_model_cnn\train'
num_classes = 7
batch_size = 64
num_epochs = 10
learning_rate = 0.001
best_model_path = 'best_model_mini_x.pth'

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Transform ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ✅ 1-channel grayscale input
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # ✅ Normalize for grayscale
])

# --- Dataset ---
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
print(f"Detected classes: {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

# --- Model ---
model = create_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

# --- Training Loop ---
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    start_time = time.time()

    model.train()
    train_loss = 0.0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    avg_train_loss = train_loss / len(train_dataset)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(f"Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # --- Save Best ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved with Val Accuracy: {val_acc:.4f}")

    print(f"⏱️ Time per epoch: {(time.time() - start_time):.2f}s")

print(f"\n🎯 Training finished. Best Val Accuracy: {best_val_acc:.4f}")
