import cv2
import numpy as np

def process_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = (256,256)
    interpolation_method = cv2.INTER_LANCZOS4
    img = cv2.resize(img, size, interpolation_method) 
    img = img.astype(float)/255.0
    img = np.expand_dims(np.array(img), axis = (0,-1))
    return img