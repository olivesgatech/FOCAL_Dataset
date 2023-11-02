from Models.detection.yolov3.models import Darknet
from Models.detection.yolov5.models import yolo, common
from Models.detection.yolov5.utils.downloads import attempt_download
from Models.detection.yolov5.utils.general import intersect_dicts
import torch
from config import BaseConfig


def build_detection_architecture(cfg: BaseConfig):
    if cfg.detection.model == 'yolov3':
        return Darknet(cfg=cfg)
    elif cfg.detection.model == 'yolov5':
        if cfg.detection.pretrained.endswith('.pt'):
            weights = attempt_download(cfg.detection.pretrained)
            ckpt = torch.load(weights, map_location='cpu')
            ckpt['model'].yaml['nc'] = cfg.data.num_classes
            model = yolo.DetectionModel(cfg=ckpt['model'].yaml)
            exclude = ['anchor']
            csd = ckpt['model'].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            return model
        else:
            return yolo.DetectionModel(cfg=cfg.detection.yolocfg)
    else:
        raise Exception('Model not implemented yet')
