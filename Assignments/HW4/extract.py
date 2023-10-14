import pickle
import os
import numpy as np
from PIL import Image

# Path to the CIFAR-10 batches directory
cifar10_dir = 'cifar-10-python.tar/cifar-10-python/cifar-10-batches-py'
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images

def save_images_from_batch(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'image_{idx}.png'))

# Load and save each batch
for batch_file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
    batch_images = load_batch(os.path.join(cifar10_dir, batch_file))
    save_images_from_batch(batch_images, batch_file)
