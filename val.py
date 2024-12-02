import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("pneumonia_detection_model.h5") 
 # Make sure the file name matches

# Function to make predictions on new images
def predict_pneumonia(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Use the size your model expects
    img_array = image.img_to_array(img) / 255.0  # Normalize if your model was trained with normalized data
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the loaded model
    prediction = model.predict(img_array)

    # Assuming binary classification: 0 for 'Normal', 1 for 'Pneumonia'
    if prediction[0][0] > 0.5:
        print("Prediction: Pneumonia")
    else:
        print("Prediction: Normal")

# Example usage
predict_pneumonia("virus.jpeg")