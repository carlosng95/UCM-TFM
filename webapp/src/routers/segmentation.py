from fastapi import APIRouter,Request
from algorithms.segmentation.processing.segmentation_preprocessing import dice_loss, dice_coef, apply_mask
from fastapi.templating import Jinja2Templates
import os 
import keras
import cv2
import base64


current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.abspath(os.path.join(current_dir, '..', 'public', 'templates'))
templates = Jinja2Templates(directory=templates_dir)


router = APIRouter(
    prefix = '/segmentation',
    tags = ['Segmentation']
)


@router.post("/")
def segmentation(request: Request):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'segmentation', 'models', 'seg_cnn_1.h5')
        paths = os.path.join(os.path.dirname(__file__), '..', 'public', 'img','tumors')
        images_ = [x.split('timestamp')[0] for x in os.listdir(paths)]
        if(len(images_) > 0):
            idx = images_.index(max(images_))
            img_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img', 'tumors', os.listdir(paths)[idx])
            model = keras.models.load_model(model_path, custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef})
            final_img = apply_mask(img_path, model)
            final_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img','segmented','segmented_' + os.listdir(paths)[idx])
            cv2.imwrite(final_path, final_img)
            msg = 'Tumor segmented!'
            with open(final_path, 'rb') as f:
                contents = f.read()
            base64_encoded_image = base64.b64encode(contents).decode("utf-8")
            os.remove(final_path)
            os.remove(img_path)
            return templates.TemplateResponse("segmentation_tumor.html", {"request": request,  "image": base64_encoded_image, "condition": msg})
        else:
            paths = os.path.join(os.path.dirname(__file__), '..', 'public', 'img')
            images_ = [x.split('timestamp')[0] for x in os.listdir(paths)]
            main_path =os.path.join(os.path.dirname(__file__), '..', 'public', 'img')
            files = os.listdir(main_path)
            files_ = [x.split('timestamp')[0] for x in files]
            images = [x for x in files_ if x.isdigit()]
            images = max(images)
            idx = files_.index(images)
            img_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'img', files[idx])
            with open(img_path, 'rb') as f:
                contents = f.read()
            base64_encoded_image = base64.b64encode(contents).decode("utf-8")
            os.remove(img_path)
            
            return templates.TemplateResponse("segmentation_healthy.html", {"request": request,  "image": base64_encoded_image, "condition": ''})