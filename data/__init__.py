from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from data.loader.detection.yolodata import YOLOFrameDataset, YOLOv5FrameDataset
from data.utils.customtransforms import Cutout
from trainutils.data.dataobjects import LoaderObject, DatasetStructure
from config import BaseConfig



def get_yolo_dataset(cfg: BaseConfig, idxs: np.ndarray = None, debug: bool = False):
    # add transforms
    augmentations = []

    if cfg.data.augmentations.random_hflip:
        augmentations.append(A.HorizontalFlip())
    if cfg.data.augmentations.random_crop:
        augmentations.append(A.RandomCrop(width=cfg.detection.img_size, height=cfg.detection.img_size))
    augmentations.append(A.Resize(cfg.detection.img_size, cfg.detection.img_size))
    mean = [0, 0, 0]
    std = [1, 1, 1]
    augmentations.append(A.Normalize(mean=mean, std=std, max_pixel_value=255, ))
    augmentations.append(ToTensorV2())

    train_transform = A.Compose(augmentations, bbox_params=A.BboxParams(format="yolo", min_area=0.0,
                                                                        label_fields=[]))
    # test transforms
    test_transform = A.Compose(
        [
            A.Resize(cfg.detection.img_size, cfg.detection.img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=0.0, label_fields=[]),
    )
    
    data_tr = YOLOv5FrameDataset(split='train', cfg=cfg, transform=train_transform, current_idxs=idxs, debug=debug)
    data_te = YOLOv5FrameDataset(split='test', cfg=cfg, transform=test_transform, debug=debug)

    # data_tr = YOLOFrameDataset(split='train', cfg=cfg, transform=train_transform, current_idxs=idxs, debug=debug)
    # data_te = YOLOFrameDataset(split='test', cfg=cfg, transform=test_transform, debug=debug)

    # init data configs
    data_config = DatasetStructure(cfg=cfg)
    data_config.train_set = data_tr
    data_config.test_set = data_te
    data_config.train_len = len(data_tr)
    data_config.test_len = len(data_te)
    # TODO: automate the number of classes to label compression
    data_config.num_classes = cfg.data.num_classes
    data_config.img_size = cfg.detection.img_size
    data_config.in_channels = cfg.data.in_channels

    data_config.is_configured = True

    # tr_loader = DataLoader(data_tr,
    #                        batch_size=cfg.detection.batch_size,
    #                        shuffle=True)
    # te_loader = DataLoader(data_te,
    #                        batch_size=cfg.detection.batch_size,
    #                        shuffle=False)

     # For yolov5 yash added this since each image has different number of objects a
     # collate function is needed in data loader
    tr_loader = DataLoader(data_tr,
                           batch_size=cfg.detection.batch_size,
                           shuffle=True, collate_fn=YOLOv5FrameDataset.collate_fn)
    te_loader = DataLoader(data_te,
                           batch_size=cfg.detection.test_batch_size,
                           shuffle=False, collate_fn=YOLOv5FrameDataset.collate_fn)

    loaders = LoaderObject(train_loader=tr_loader,
                           test_loader=te_loader,
                           val_loader=tr_loader,
                           data_configs=data_config)

    return loaders

