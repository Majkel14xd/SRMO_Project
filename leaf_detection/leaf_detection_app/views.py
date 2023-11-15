from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
from io import BytesIO
import cv2
import tensorflow as tf
import numpy as np
import base64
import h5py
import json
# Create your views here.
def index(request):
      if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image
        uploaded_image = request.FILES['image']
        image_path = default_storage.save('images_tmp/uploaded_image.jpg', ContentFile(uploaded_image.read()))
        # Load the model
        model_path = "leaf_detection_app/models/leaf_classifier.h5"
        model = tf.keras.models.load_model(model_path)

        # Load class names
    

        # Load and preprocess the image using cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        target_size = (256, 256) 
        image = cv2.resize(image, target_size)
        input_arr = tf.keras.utils.img_to_array(image) / 255.0
        input_arr = np.array([input_arr])
        with h5py.File(model_path,'r') as fp:
            class_names = json.loads(fp.attrs.get('class_names'))
        # Make predictions
        predictions = model.predict(input_arr)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(100 * (np.max(tf.nn.softmax(predictions))), 2)

        # Create a PIL Image from the numpy array
        pil_image = Image.fromarray((input_arr[0] * 255).astype(np.uint8))

        # Save the PIL Image to BytesIO
        image_io = BytesIO()
        pil_image.save(image_io, format='JPEG')

        # Convert BytesIO to base64 string for HTML embedding
        image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')


        # Log the base64 image for debugging
        print("Base64 Image:", image_base64)

        context = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_base64': image_base64,
            'uploaded_image' : uploaded_image,
        }
        return render(request, 'index.html', context)
      
      return render(request, 'index.html')