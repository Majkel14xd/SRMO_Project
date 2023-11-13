from PIL import Image
import os

def convert_tif_to_jpg(input_folder, output_folder, target_size=(100, 200)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(root, file)           
                tif_image = Image.open(tif_path)
                
                # Resize the image to the target size
                resized_image = tif_image.resize(target_size)
                
                jpg_path = os.path.join(output_folder, os.path.relpath(tif_path, input_folder)).replace(".tif", ".jpg")
                jpg_folder = os.path.dirname(jpg_path)
                if not os.path.exists(jpg_folder):
                    os.makedirs(jpg_folder)
                    
                # Save the resized image as JPEG
                resized_image.convert("RGB").save(jpg_path, "JPEG")
                
                print(f"Skonwertowano: {tif_path} -> {jpg_path}")

input_folder_path = "images"
output_folder_path = "images_jpg"
convert_tif_to_jpg(input_folder_path, output_folder_path)
