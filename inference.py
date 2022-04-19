from pathlib import Path

import torch
from easydict import EasyDict as edict

from models.common import AutoShape, DetectMultiBackend

params = edict(
    # model_name='yolov5s',
    weight_path='weights/best.pt',
    device='gpu',
    conf_thres=0.5,
    iou_thres=0.45,
)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_model(path='yolov5s', check_point=None, conf_thres=0.25, iou_thres=0.45, device='gpu'):
    '''
    '''
    # cfg = list((Path(__file__).parent / 'models').rglob(f'{path}.yaml'))[0]  # model.yaml path
    _device = torch.device('cuda' if device == 'gpu' else 'cpu')
    model = DetectMultiBackend(path, device=_device)
    # model = Model(cfg, params.channels, params.classes)  # create model
    if check_point:
        ckpt = torch.load(check_point)  # load
        msd = model.state_dict()  # model state_dict
        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        csd = {k: v for k, v in csd.items(
        ) if msd[k].shape == v.shape}  # filter
        model.load_state_dict(csd, strict=False)  # load
        if len(ckpt['model'].names) == params.classes:
            model.names = ckpt['model'].names  # set class names attribute

    # for file/URI/PIL/cv2/np inputs and NMS
    model = AutoShape(model, conf_thres, iou_thres)
    if device == 'gpu':
        model = model.cuda()
    return model


def inference(model, imgs):
    model = model.eval()
    results = model(imgs)

    return results


def preprocess(img):
    return img


def postprocess(results):
    files = results.files
    xyxyn = results.xyxyn
    classes = results.names
    imgs = results.imgs
    total_res = []
    for i in range(len(files)):
        img_path = files[i]
        h, w, _ = imgs[i].shape
        xyxyni = xyxyn[i].cpu().numpy()
        res = []
        for res_one in xyxyni:
            res.append(dict(
                xyxy=list(map(float, res_one[:4].tolist())),
                conf=float(res_one[4]),
                class_id=int(res_one[5]),
                class_name=classes[int(res_one[5])],
            ))
        total_res.append(dict(
            img_name=Path(img_path).name,
            img_size=(w, h),
            objects=res,
        ))

    return total_res


def save_result(results, save_path):
    import json
    from pathlib import Path
    save_path = Path(save_path)
    with save_path.open('w') as f:
        json.dump(results, f)


def get_info(res):
    objs = res['objects']
    faces = []
    for obj in objs:
        if obj['class_name'] == 'face':
            faces.append(obj)

    info = dict(
        one_face=False,
        with_mask=False,
        with_hat=False,
    )
    is_uniface = len(faces) == 1
    if is_uniface:
        info['one_face'] = True
        for obj in objs:
            if obj['class_name'] == 'mask':
                info['with_mask'] = True
            if obj['class_name'] == 'hat':
                info['with_hat'] = True

    return info


if __name__ == '__main__':
    img_path = './images/face/1076.jpg'
    save_dir = './results'

    img_path = Path(img_path)
    save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True)

    if img_path.is_file():
        imgs = [img_path.__str__()]
    elif img_path.is_dir():
        imgs = [path.__str__() for path in img_path.glob('*')]

    print('loading model...')
    model = get_model(path=params.weight_path,
                      #   check_point=params.weight_path,
                      device=params.device,
                      conf_thres=params.conf_thres,
                      iou_thres=params.iou_thres)

    print('preprecessing...')
    imgs = preprocess(imgs)

    print('inferencing...')
    results = inference(model, imgs)

    print('postprecessing...')
    results_preced = postprocess(results)

    # save_result(total_res, 'result.json')

    img_info = get_info(results_preced[0])
    # print(f'results saved in {save_dir}')
