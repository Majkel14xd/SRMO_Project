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
    image_size = (100, 200),  
    seed       = 1337,
)
class_names = dataset.class_names
class_names
sample_batch = dataset.take(1)
for image_batch, label_batch in sample_batch:
    print(image_batch.shape)
    print(label_batch.numpy())
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
image_path = "test_images\Brzoza_test.jpg"
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
confidence = round(100 * (np.max(tf.nn.softmax(predictions))), 2)

# Display the results
plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
plt.imshow(input_arr[0])  # Display the normalized image
plt.axis("off")
plt.show()

model.summary()

# Get the list of layers in the model
layers_list = model.layers

# Iterate through each layer and display the number of weights
for layer in layers_list:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    if weights:
        print(f"Number of weights: {len(weights)}")
        for i, weight_array in enumerate(weights):
            print(f"  Weight {i + 1} shape: {weight_array.shape}")
    else:
        print("No weights in this layer")
    print("\n")

    plt.figure(figsize=(20, 20))

# take a batch from 'test_ds'
for image_batch, label_batch in test_dataset.take(1):
    
    # returns array of confidences for different classes of all images in the batch
    prediction_batch = model.predict(image_batch)

    size = len(image_batch)
    columns = 4
    rows = size//columns
    
    for i in range(size):
        
        image = image_batch[i].numpy().astype("uint8")   # converting float to int
        
        actual_class    = class_names[label_batch[i]]
        predicted_class = class_names[np.argmax(prediction_batch[i])]
        confidence      = round(100 * (np.max(prediction_batch[i])), 2)
        
        ax = plt.subplot(rows, columns, i + 1)   # row, col, idx

        title = plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.setp(title, color= 'g' if actual_class == predicted_class else 'r')
        plt.imshow(image)
        plt.axis("off")