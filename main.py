import cv2
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

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