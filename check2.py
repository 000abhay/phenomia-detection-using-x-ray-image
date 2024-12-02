# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set parameters
input_shape = (224, 224, 3)
batch_size = 32
epochs = 20

# Load MobileNetV2 with pre-trained weights, without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False  # Freeze the base model

# Define the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Only rescaling for validation and test data
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load train, validation, and test data
train_data = train_datagen.flow_from_directory(
   'C:/Users/user/Desktop/archive/chest_xray/train',  # Replace with your actual path for training data
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

validation_data = val_test_datagen.flow_from_directory(
    'C:/Users/user/Desktop/archive/chest_xray/val',  # Replace with your actual path for validation data
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    'C:/Users/user/Desktop/archive/chest_xray/test',  # Replace with your actual path for test data
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Don't shuffle test data to get consistent results
)

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('C:/Users/user/Documents/pneumonia_detection_model.h5')  # Replace with your actual path to save the model