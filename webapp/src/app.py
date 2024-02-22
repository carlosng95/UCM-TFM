import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import JSONResponse,HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from routers import image, detection, segmentation
from fastapi.templating import Jinja2Templates
import shutil
import os
import uvicorn

app = FastAPI(
    title = 'TFM: Detection and Segmentation of Brain Tumor',
    description = 'Mi nombre',
    version = '1.0.0'
)

templates = Jinja2Templates(directory = './public/templates')

app.mount('/static',StaticFiles(directory = './public/static'), name = 'static')
# app.mount('/img', StaticFiles(directory = './public/img'), name = 'img')

@app.get('/', response_class=HTMLResponse)
def root():
    path = './public/static/views/index.html'
    img_path = './public/img'
    try:
        shutil.rmtree(img_path)
    except:
        pass
    os.mkdir(img_path)
    return FileResponse(path, status_code = 200)


app.include_router(image.router)
app.include_router(detection.router)
app.include_router(segmentation.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)