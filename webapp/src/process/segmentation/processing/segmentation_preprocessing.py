import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def preprocess(data):
    data = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in data]
    data = [cv2.resize(x, (256,256)) for x in data]
    data = [image.astype(float)/255.0 for image in data]
    return data

    

def plot_image(data,idx, masks = None):
    imagen = data[idx]
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(imagen, cmap='gray')
    if len(masks) > 0:
        ax.imshow(masks[idx], alpha=0.3, origin='lower')
        
def get_one_prediction(img, model):
    img = np.expand_dims(img, axis=(0, -1))
    return model.predict(img)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def apply_mask(path,model):
    img = cv2.imread(path)
    img_p = preprocess([img])[0]
    mask = get_one_prediction(img_p, model)
    mask = np.squeeze(mask)
    mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.merge([img_p, img_p, img_p])*255.0
    mask = mask.astype(img.dtype)
    final_img = cv2.addWeighted(img, 0.8, mask, 0.2, 0)
    return final_img