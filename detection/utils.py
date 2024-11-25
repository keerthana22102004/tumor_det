import cv2
import numpy as np
import tensorflow as tf
from django.core.files.uploadedfile import InMemoryUploadedFile

# Load the pre-trained VGG-16 model
model_path = r'C:\Users\gskee\OneDrive\文档\INTERNSHIP\INFOSYS\iris_tumor_detection_git 2\detection\iris_tumor_model_vgg16.keras'
cnn_model = tf.keras.models.load_model(model_path)

# Image dimensions expected by the VGG-16 model
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Preprocessing function for VGG-16
def preprocess_image(image_file: InMemoryUploadedFile):
    # Convert uploaded file to OpenCV format
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Check if the image is valid
    if img is None:
        raise ValueError("Invalid image provided")
    
    # Resize and normalize the image to the VGG-16 input size
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = img_resized / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

# Detection function
def detect_tumor(image_file: InMemoryUploadedFile):
    img_array = preprocess_image(image_file)
    
    # Predict using the CNN model
    prediction = cnn_model.predict(img_array)[0][0]  # Prediction score
    
    # Determine the label and confidence
    class_label = "TUMOR DETECTED" if prediction >= 0.5 else "NO TUMOR DETECTED"
    
    return class_label

# Example usage in a Django view
def handle_uploaded_image(image_file: InMemoryUploadedFile):
    try:
        label= detect_tumor(image_file)
        return f"{label}"
    except ValueError as e:
        return str(e)
