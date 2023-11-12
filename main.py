import cv2
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os, json, h5py
from datetime import datetime
from sklearn.model_selection import train_test_split

dataset = tf.keras.utils.image_dataset_from_directory(
    directory  = "./images_jpg", 
    color_mode = 'rgb',     
    batch_size = 32,
    image_size = (100, 250),  
    seed       = 1337,
)

# Podział na dane treningowe i testowe
train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size).take(test_size)
# Normalizacja danych
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

# Budowa modelu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(dataset.class_names))  # liczba klas równa liczbie rodzajów liści
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trenowanie modelu
# Zmniejszamy liczbę epok do 5, aby uniknąć przetrenowania na małym zestawie danych
history = model.fit(normalized_train_dataset, epochs=5, validation_data=test_dataset)

# Zapisanie modelu
model.save('leaf_classifier.h5')

model_path = "leaf_classifier.h5"
model = tf.keras.models.load_model(model_path)

if hasattr(model, 'class_names'):
    class_names = model.class_names
class_names


# Load the image
image_path = "test_images/brzoza_test.jpg"
image = tf.keras.utils.load_img(image_path)

# Resize the image to match the model's expected input shape
target_size = (100, 200)
image = tf.image.resize(np.array(image), target_size)

# Convert the resized image to an array and normalize pixel values
input_arr = tf.keras.utils.img_to_array(image) / 255.0
input_arr = np.array([input_arr])  # Convert single image to a batch.

# Make predictions
predictions = model.predict(input_arr) 
predicted_class = class_names[np.argmax(predictions)]
confidence = round(100 * (np.max(tf.nn.softmax(prediction_batch[i]))), 2)

# Display the results
plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
plt.imshow(input_arr[0])  # Display the normalized image
plt.axis("off")
plt.show()
