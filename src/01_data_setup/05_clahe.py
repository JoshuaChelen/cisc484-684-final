import os
import cv2
import numpy as np
from PIL import Image

#paths
INPUT_DIR  = "data/HAM10000_images_resized"
OUTPUT_DIR = "data/HAM10000_images_clahe"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#create clahe
clahe=cv2.createCLAHE