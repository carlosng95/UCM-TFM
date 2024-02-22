from typing import List, Optional
from fastapi import status,HTTPException, Depends, APIRouter, Response,File, UploadFile, Request
from fastapi.responses import JSONResponse,FileResponse 
from fastapi.templating import Jinja2Templates
import os 
import shutil
from datetime import datetime
import base64
import cv2


current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.abspath(os.path.join(current_dir, '..', 'public', 'templates'))
templates = Jinja2Templates(directory=templates_dir)

router = APIRouter(
    prefix = '/image',
    tags = ['Image']
)


@router.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        target_folder = "../src/public/img"
        os.makedirs(target_folder, exist_ok=True)
        idx = str(round(datetime.now().timestamp()*1000))
        filename = idx+'timestamp'+file.filename
        extension = filename.split('.')[-1]
        contents = file.file.read()
        with open(os.path.join(target_folder, filename), "wb") as f:
            f.write(contents)
        if extension != 'jpg':
            img = cv2.imread(os.path.join(target_folder, filename))
            filename_ = '.'.join(filename.split('.')[:-1])+'.jpg'
            cv2.imwrite(os.path.join(target_folder, filename_),img)
            os.remove(os.path.join(target_folder, filename))
            with open(os.path.join(target_folder, filename_), 'rb') as f:
                contents = f.read()
        else: 
            pass
    except Exception:
        path = './public/static/views/index.html'
        return FileResponse(path, status_code = 200)

    finally:
        file.file.close()
        f.close()
    base64_encoded_image = base64.b64encode(contents).decode("utf-8")
    return templates.TemplateResponse("image.html", {"request": request,  "image": base64_encoded_image})
    

