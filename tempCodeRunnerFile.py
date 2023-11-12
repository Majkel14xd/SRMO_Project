import cv2
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
import numpy as np
import matplotlib.pyplot as plt

dataset = tf.keras.utils.image_dataset_from_directory(
    directory  = "./images_jpg", 
    color_mode = 'rgb',     
    batch_size = 32,
    image_size = (256, 256),  
    shuffle    = True,
    seed       = None,
    crop_to_aspect_ratio = False,
)


class_names = dataset.class_names
print(class_names)

sample_batch = dataset.take(1)
print(sample_batch)

for image_batch, label_batch in sample_batch:
    print(image_batch.shape)
    print(label_batch.numpy())


plt.figure(figsize=(20, 12))
for image_batch, label_batch in sample_batch:
    size = len(image_batch)
    columns = 8
    rows = size//columns
    for i in range(size):
        ax = plt.subplot(rows, columns, i + 1)   # rows, cols, idx
        plt.title(class_names[label_batch[i]])
        plt.imshow(image_batch[i].numpy().astype("uint8"))   # converting float to int
        plt.axis("off")

def split_dataset(ds, ratios=(8, 1, 1,)):
    total = sum(ratios)
    ds_size = len(ds)
    skip = 0
    for ratio in ratios[:-1]:
        size = int(ds_size * ratio/total)
        yield ds.skip(skip).take(size)
        skip += size
    yield ds.skip(skip)

train_ds, val_ds, test_ds = split_dataset(dataset, (240, 16, 1))

# ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

print("Total:", len(dataset))
print("Train:", len(train_ds))
print("Valid:", len(val_ds))
print("Test :", len(test_ds))


for image_batch, label_batch in sample_batch:
    input_shape = image_batch.shape

model = models.Sequential([

    # preprocessing
    layers.experimental.preprocessing.Resizing(256, 256),   # resize images
    layers.experimental.preprocessing.Rescaling(1./255),    # normalize
    
    # data augmentation
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=None),
    layers.experimental.preprocessing.RandomRotation(0.5, seed=None),
    layers.RandomContrast(0.3, seed=None),
    layers.RandomZoom(0.3, seed=None),
    
    # layering
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # flatten
    layers.Flatten(),
    
    # dense layer
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax'),
])

model.build(input_shape=input_shape)
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

keras_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
    #   callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min'),
]

history = model.fit(
    train_ds,
    validation_data  = val_ds,
    validation_split = 0.0,     # validation_data gets priority
    batch_size       = None,    # defaults to 32
    verbose          = 'auto',  # 0=silent, 1=progress bar, 2=one line per epoch
    epochs           = 10,
    callbacks        = keras_callbacks
)