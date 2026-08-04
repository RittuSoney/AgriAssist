import os
import numpy as np
import tensorflow as tf
from PIL import Image
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# 1. Loaded the model globally so it only loads ONCE when the server starts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'saved_models', 'agriassist_base_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Hardcoded the 38 class names here EXACTLY as they appeared in Colab
CLASS_NAMES = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def home(request):
    return render(request, 'upload.html')

@api_view(['POST'])
def predict_disease(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=400)

    # Saved the uploaded image temporarily
    uploaded_file = request.FILES['image']
    fs = FileSystemStorage()
    filename = fs.save(uploaded_file.name, uploaded_file)
    absolute_path = fs.path(filename)

    try:
        # Preprocessing the image for MobileNetV2
        img = Image.open(absolute_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 224, 224, 3)

        # Making the prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = round(float(np.max(predictions[0])) * 100, 2)

        result = {
            'disease': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence
        }

    except Exception as e:
        return Response({'error': str(e)}, status=500)
    finally:
        # Cleaning up the image from the hard drive after prediction
        if os.path.exists(absolute_path):
            os.remove(absolute_path)

    return Response(result)
