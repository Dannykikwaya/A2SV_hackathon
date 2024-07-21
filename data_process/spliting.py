import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(input_folder, output_folder, test_size=0.1, val_size=0.2):
    images = os.listdir(input_folder)
    train_val_images, test_images = train_test_split(images, test_size=test_size)
    train_images, val_images = train_test_split(train_val_images, test_size=val_size)
    
    def copy_images(images, subset):
        subset_folder = os.path.join(output_folder, subset)
        if not os.path.exists(subset_folder):
            os.makedirs(subset_folder)
        for img in images:
            shutil.copy(os.path.join(input_folder, img), subset_folder)
    
    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    copy_images(test_images, 'test')

split_data('chemin_vers_dossier_input', 'chemin_vers_dossier_output')
