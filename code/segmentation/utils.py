import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interactive, IntSlider, fixed
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, InputLayer, Flatten,Activation, Conv2DTranspose, Concatenate, Input
from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


def load_images(paths, data):
    images = []
    bbox = []
    for image_file in paths:
        file = image_file.split('/')[-1]
        id = list(filter(lambda x: x['file_name'] == file, data['images']))[0]['id']
        if len(list(filter(lambda x: x['image_id'] == id, data['annotations']))) > 0:
            box = list(filter(lambda x: x['image_id'] == id, data['annotations']))[0]['bbox']
            images.append(cv2.imread(image_file))
            bbox.append((id,box))

    bbox = pd.DataFrame(bbox, columns = ['id','bbox'])
    return images, bbox

def image_preprocess(data):
    data = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in data]
    data = [cv2.resize(x, (256,256)) for x in data]
    data = [image.astype(float)/255.0 for image in data]
    return data

def image_mask_preprocess(data):
    y = data['bbox']
    y =[[x*256/640 for x in q] for q in y]
    return y
    

def create_single_mask(A, width, height):
    A = list(map(int, A))
    a, b, w, h = A[0], A[1], A[2], A[3]
    mask = np.zeros((height, width))  
    mask[b:b+h, a:a+w] = 1  
    return mask

def create_masks(data):
    return [create_single_mask(x, 256,256) for x in data]   



def plot_image(data,idx, masks = None):
    imagen = data[idx]
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(imagen, cmap='gray')
    if len(masks) > 0:
        ax.imshow(masks[idx], alpha=0.3, origin='lower')
    
def image_slider(X, masks= None):
    idx_slider = IntSlider(value=0, min=0, max=len(X)-1, step=1, description='index')
    if len(masks) > 0:
        return interactive(plot_image, idx=idx_slider, data = fixed(X), masks = fixed(masks))
    else:
        return interactive(plot_image, idx=idx_slider, data = fixed(X))
    
def convolutional_block(input, filters):
    x = Conv2D(filters, 3, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, 3, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def enconder_block(input, filters):
    x = convolutional_block(input, filters)
    pool = MaxPool2D((2,2))(x)
    return x,pool

def decoder_block(input, skip_features, filters):
    x = Conv2DTranspose(filters, 2, strides = 2, padding = 'same')(input)
    x = Concatenate()([x, skip_features])
    x = convolutional_block(x, filters)
    return x

def unet(input_shape):
    inputs = Input(input_shape)
    
    s1, p1 = enconder_block(inputs, 64)
    s2, p2 = enconder_block(p1, 128)
    s3, p3 = enconder_block(p2, 256)
    s4, p4 = enconder_block(p3, 512)
    
    b1 = convolutional_block(p4, 1024)
    
    d1 = decoder_block(b1, s4,  512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    outputs = Conv2D(1,1, padding = 'same', activation = 'sigmoid')(d4)
    
    model = Model(inputs, outputs, name = 'UNET')
    return model


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def get_one_prediction(img, model):
    img = np.expand_dims(img, axis=(0, -1))
    return model.predict(img)

def get_set_prediction(data, model):
    return model.predict(np.expand_dims(data,-1))

def area(masks):
    return [x.sum() for x in masks]
    
def get_metrics(masks, pred):
    prods = [x*y[:,:,0] for x,y in zip(np.array(masks),pred)]
    areas = [(x.sum(), y.sum(), z.sum()) for x,y,z in zip(np.array(masks), prods, pred)]
    areas = pd.DataFrame(areas, columns = ['Actual','Common', 'Predicted'])
    areas['Dice_coeff'] = 2*areas['Common']/(areas['Predicted'] + areas['Actual'])
    areas['Ratio'] = areas['Common']/areas['Actual']
    return areas