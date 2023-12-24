import os
import base64
import numpy as np
from django.shortcuts import render
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch
from django.conf import settings
import cv2

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        subdirectory = 'uploaded_images'  
        image_path = default_storage.save(os.path.join(subdirectory, uploaded_image.name), ContentFile(uploaded_image.read()))
        num_classes = ['acer', 'alnus_incana', 'betula_pubescens', 'facus_silvatica', 'populus', 'populus_tremula', 'quercus', 'salix_aurita', 'salix_sinerea', 'sericea', 'sorbus_aucuparia', 'sorbus_intermedia', 'tilia', 'ulmus_carpinifolia', 'ulmus_glabra']
        num_classes_pl = ['klon', 'olsza_szara', 'brzoza_omszona', 'buk_zwyczajny', 'topola', 'topola_osika', 'dab', 'wierzba_uszata', 'wierzba_siwa', 'wierzba', 'jarzab_zwyczajny', 'jarzab_srodkowy', 'lipa', 'wiaz_klosowy', 'wiaz_gladki']
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(num_classes))
        model.load_state_dict(torch.load('leaf_detection_app/models/model_leaf_detection.pth'))
        model.eval()

        image = Image.open(os.path.join(settings.MEDIA_ROOT, image_path))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = num_classes[predicted_idx.item()]
            predicted_class_pl = num_classes_pl[predicted_idx.item()]
        image_np = np.array(image)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render(request, 'index.html', {'image_base64': image_base64, 'prediction': predicted_class,'prediction_pl': predicted_class_pl,'file_name': uploaded_image.name})
    
    return render(request, 'index.html')
