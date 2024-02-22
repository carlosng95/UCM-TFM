from typing import List, Optional
from fastapi import status,HTTPException, Depends, APIRouter, Response,File, UploadFile, Request
from fastapi.responses import JSONResponse,  FileResponse
from process.detection.processing.detection_preprocessing import process_image
from fastapi.templating import Jinja2Templates
import os 
import numpy as np
import keras
import cv2
import base64



current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.abspath(os.path.join(current_dir, '..', 'public', 'templates'))
templates = Jinja2Templates(directory=templates_dir)



router = APIRouter(
    prefix = '/detection',
    tags = ['Detection']
)

@router.post("/")
def prediction(request: Request):
    tumor_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img','tumors')
    if not os.path.exists(tumor_path):
        os.makedirs(tumor_path)
    else:
        pass
    
    segmented_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img','segmented')
    if not os.path.exists(segmented_path):
        os.makedirs(segmented_path)
    else:
        pass
        
    model_path = os.path.join(os.path.dirname(__file__), '..', 'process', 'detection', 'models', 'cnn_1.h5')
    paths = os.path.join(os.path.dirname(__file__), '..', 'public', 'img')
    images_ = [x.split('timestamp')[0] for x in os.listdir(paths)]
    if(len(images_) > 2):
        main_path =os.path.join(os.path.dirname(__file__), '..', 'public', 'img')
        files = os.listdir(main_path)
        files_ = [x.split('timestamp')[0] for x in files]
        images = [x for x in files_ if x.isdigit()]
        images = max(images)
        idx = files_.index(images)
        img_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img', files[idx])     
        model = keras.models.load_model(model_path)

        with open(img_path, 'rb') as f:
                contents = f.read()
                
        img = process_image(img_path)
        pred = model.predict(img)[0][0]
        if pred >= 0.5:
            msg = 'Tumor'
        else:
            msg = 'Healthy'
            
        if msg == 'Tumor':   
            tumor_path += '/' + os.listdir(paths)[idx]
            tumor_path = '.'.join(tumor_path.split('.')[:-1])+'.jpg'
            img = cv2.imread(img_path)
            cv2.imwrite(tumor_path, img)
        #os.remove(img_path)
        base64_encoded_image = base64.b64encode(contents).decode("utf-8")
        return templates.TemplateResponse("detection.html", {"request": request,  "image": base64_encoded_image, "condition": msg})
    else:
        msg = 'There is not image to process!'
        
    path = './public/static/views/index.html'
    return FileResponse(path, status_code = 200)