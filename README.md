# 🌱 AgriAssist AI

AgriAssist AI is a full-stack web application designed to act as a digital plant health scanner. By uploading a clear image of a plant leaf, the application leverages a Deep Learning model to instantly diagnose crop diseases and provide a confidence score.

## 🚀 Features

* **Instant Machine Learning Inference:** Uses a MobileNetV2 Convolutional Neural Network (CNN) backend to process image arrays and classify plant diseases.
* **Out-of-Distribution Rejection:** Built-in 80% confidence thresholding to gracefully reject unrecognizable images or non-plant inputs.
* **Seamless API Integration:** Vanilla JavaScript `fetch` API implementation that communicates with the Django backend asynchronously, secured by dynamic CSRF tokens.
* **Custom UI/UX:** A responsive, modern frontend built entirely with vanilla CSS (no external libraries). Features an intuitive drag-and-drop zone, dynamic progress bars, and an earthy color palette utilizing custom CSS variables.
* **Automated Data Parsing:** Backend logic that splits complex model output strings (e.g., `Strawberry_Leaf_Scorch`) into clean, user-friendly UI elements (`Crop: Strawberry`, `Disease: Leaf Scorch`).

## 🛠️ Tech Stack

* **Frontend:** HTML5, Vanilla CSS3, Vanilla JavaScript
* **Backend:** Python, Django
* **Machine Learning:** TensorFlow / Keras (MobileNetV2)
* **Image Processing:** NumPy, Pillow (PIL)

## 📁 Project Structure

```text
agriassist_backend/
├── agri_core/               # Main Django project settings & routing
├── disease_predictor/       # Core Django app handling the ML prediction API
│   ├── views.py             # Image preprocessing, thresholding, and model inference
│   └── urls.py              # API endpoint routing
├── templates/               
│   └── upload.html          # Custom styled frontend interface
├── saved_models/            # Directory containing the trained .keras model
├── manage.py
└── .env                     # Hidden environment variables (SECRET_KEY)

## 💻 Local Installation Guide

Clone the repository:
   ```bash
    git clone [https://github.com/RittuSoney/AgriAssist_AI.git](https://github.com/RittuSoney/AgriAssist_AI.git)
    cd AgriAssist_AI/agriassist_backend