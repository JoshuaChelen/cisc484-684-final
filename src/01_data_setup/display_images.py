import os 
import random
import numpy as np
from PIL import Image


#save 5 random images from the output path
def display_images(path, operation):
    images = [
        f for f in os.listdir(path)
        if f.endswith((".jpg", ".jpeg", ".png", ".npy"))
    ]
    output_folder="samples"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    iterations=min(len(images), 5)
    selected=random.sample(images, iterations)

    for i in selected:
        img_path= os.path.join(path, i)
        if img_path.endswith(".npy"):
            arr=np.load(img_path)
            if arr.max()<=1:
                arr=arr*255
            arr=arr.astype(np.uint8)
            img=Image.fromarray(arr)
            file_name=i.replace(".npy", ".jpg")
        else:
            img=Image.open(img_path)
            file_name=i
        new_path=os.path.join(output_folder, f"{operation}_sample{file_name}.jpg")
        img.save(new_path)