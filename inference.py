from pathlib import Path
import os

import cv2
from cv2 import checkHardwareSupport
import tqdm
import torch
from easydict import EasyDict as edict

from models.yolo import Model
from models.common import AutoShape


params = edict(
    model_name='yolov5m',
    weight_path='./weights/yolov5m.pt',
    channels=3,
    classes=80,
    device='cpu',
)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_model(path='yolov5s', check_point=None, device='gpu'):
    '''
    params:
        name: yolov5n, yolov5s, yolov5m or yolov5l 
    '''
    cfg = list((Path(__file__).parent /
               'models').rglob(f'{path}.yaml'))[0]  # model.yaml path

    model = Model(cfg, params.channels, params.classes)  # create model
    if check_point:
        ckpt = torch.load(check_point, map_location=device)  # load
        msd = model.state_dict()  # model state_dict
        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        csd = {k: v for k, v in csd.items(
        ) if msd[k].shape == v.shape}  # filter
        model.load_state_dict(csd, strict=False)  # load
        if len(ckpt['model'].names) == params.classes:
            model.names = ckpt['model'].names  # set class names attribute

    model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    if params.device == 'gpu':
        model = model.cuda()
    return model


def inference(model, imgs, save_dir):
    model = model.eval()
    results = model(imgs)

    return results


if __name__ == '__main__':
    img_path = './images'
    save_dir = './results'

    img_path = Path(img_path)
    save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True)

    if img_path.is_file():
        imgs = [img_path.__str__()]
    elif img_path.is_dir():
        imgs = [path.__str__() for path in img_path.glob('*')]

    print('loading model...')
    model = get_model(path=params.model_name,
                      check_point=params.weight_path, device=params.device)

    print('inferencing...')
    for img_file in imgs:
        results = inference(model, imgs, save_dir)
    print(f'results saved in {save_dir}')
    print('end.')
