# 🎯 Real-Time Emotion Detection System

This project is a real-time facial emotion detection system that uses deep learning to classify human emotions from webcam video input. It supports seven basic emotions and is optimized for lightweight inference using the **MiniXception** model.

---

## 📌 Key Features

- ✅ Real-time webcam-based emotion detection
- 🧠 MiniXception CNN for fast and efficient inference
- 📊 Trained on the FER-2013 dataset
- 🧪 Recognizes 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 🧩 Modular codebase with clear separation of model, training, and prediction logic
- ⚙️ Supports both CPU and GPU (CUDA)

---

## 📁 Project Structure

```

emotion\_model\_cnn/
├── model.py                  # Defines MiniXception model
├── train\_model.py            # Trains model on FER-2013 dataset
├── predict\_realtime.py       # Runs real-time webcam detection
├── best\_model\_mini\_x.pth     # Saved model weights
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

````

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Jayjoshi08092003/REAL_TIME_EMOTION_DETECTION.git
cd REAL_TIME_EMOTION_DETECTION
````

### 2. (Optional) Create and Activate Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### Train the Model

```bash
python train_model.py
```

> Alternatively, use the provided `best_model_mini_x.pth` file directly for prediction.

### Run Real-Time Emotion Detection

```bash
python predict_realtime.py
```

> Ensure your webcam is connected and functioning. Press `q` to exit the detection window.

---

## 🤖 Emotions Supported

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

---

## 🧠 Model Info

This project uses a **MiniXception** CNN with grayscale inputs and a `224x224` resolution for emotion classification. The model was trained on the **FER-2013** dataset using PyTorch.

---

## 📦 Dependencies

* Python 3.8+
* torch
* torchvision
* opencv-python
* numpy
* Pillow

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📬 Contact

**Author:** Jay Joshi
📧 Email: [jayjoshi08092003@gmail.com](mailto:jayjoshi08092003@gmail.com)
🔗 [GitHub](https://github.com/Jayjoshi08092003)
🔗 [LinkedIn](https://www.linkedin.com/in/jayjoshi08092003)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and build on it.

```

---

Let me know if you'd like a second version including:
- YOLOv5-based face detection
- Deployment instructions (e.g., Streamlit or Flask)
- Dataset download instructions from Kaggle

Would you like me to generate the `requirements.txt` too?
```
