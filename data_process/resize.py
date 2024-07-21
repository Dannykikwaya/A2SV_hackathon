import cv2
import os

def resize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, size)
        cv2.imwrite(os.path.join(output_folder, filename), resized_img)

resize_images('chemin_vers_dossier_input', 'chemin_vers_dossier_output')
