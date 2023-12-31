"""YOLO-based dataloader"""
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from config import BaseConfig
# from utils.augmentations import letterbox
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = {
    'Pedestrian': 0,
    'Pedestrian_With_Object': 0,
    'Wheeled_Pedestrian': 0,
    'Bicycle': 1,
    'Motorcycle': 1,
    'Car': 2,
    'Truck': 2,
    'Bus': 2,
    'Semi_truck': 2,
    'Stroller': 3,
    'Trolley': 3,
    'Cart': 3,
    'Emergency_Vehicle': 2,
    'Construction_Vehicle': 2,
    'Towed_Object': 3,
    'Vehicle_Towing': 3,
    'Trailer': 3,
}


class YOLOFrameDataset(Dataset):
    def __init__(self, split, cfg: BaseConfig, transform=None, current_idxs: np.ndarray = None, debug=False):
        cache_dir = cfg.run_configs.cache_dir
        # txt_file = os.path.expanduser(f'~/{cache_dir}/detection/yolo_{split}.txt')
        txt_file = os.path.expanduser(f'{cache_dir}{split}.txt')
        if not os.path.isfile(txt_file):
            raise Exception('You must first generate the data chache with seqdatainterface!!')

        with open(txt_file) as f:
            self.annotations = f.readlines()

        self.global_idxs = np.arange(len(self.annotations))
        self.annotations = np.array(self.annotations, dtype=object)
        self.total_samples = self.annotations.shape[0]
        if current_idxs is not None:
            self.annotations = self.annotations[current_idxs]
            self.global_idxs = self.global_idxs[current_idxs]
        self.transform = transform
        self._cfg = cfg
        self.ignore_iou_thresh = 0.5

        self._debug = True
        self.corrupted = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_and_label = self.annotations[index].strip().split(' ')
        img_and_label[0] = '/media/yash-yee/Storage/FOCAL/blurred' + img_and_label[0].split('blurred')[1]
        image = np.array(Image.open(img_and_label[0]).convert("RGB"))

        bboxes = img_and_label[1:]
        bboxes = np.array([list(map(int, box.split(',')[:-1])) for box in bboxes]).astype('float')
        # normalize the bboxes
        im_h, im_w = image.shape[:2]
        # print(bboxes)
        bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] / 2) / im_w
        bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] / 2) / im_h
        bboxes[:, 2] = bboxes[:, 2] / im_w
        bboxes[:, 3] = bboxes[:, 3] / im_h
        bboxes[:, 0:4] = np.clip(bboxes[:, :4], a_min=0.0, a_max=0.99)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], 2 * (1 - bboxes[:, 0]))
        bboxes[:, 3] = np.minimum(bboxes[:, 3], 2 * (1 - bboxes[:, 1]))

        # print(img_and_label[0])
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        bboxes = torch.tensor(bboxes)
        labels = torch.zeros((self._cfg.detection.max_detections, 5))
        if bboxes.shape[0] < self._cfg.detection.max_detections:
            upper = bboxes.shape[0]
        else:
            upper = self._cfg.detection.max_detections
        labels[:upper, 0] = bboxes[:upper, -1]
        labels[:upper, 1:] = bboxes[:upper, :-1]
        image = torch.tensor(image)
        sample = {'data': image, 'label': labels, 'idx': index, 'global_idx': self.global_idxs[index]}

        return sample

    def get_event_dict(self):
        out_dict = defaultdict(list)
        idx_to_event = []
        for i in range(self.annotations.shape[0]):
            event = self.annotations[i].split('/')[6]
            out_dict[event].append(i)
            idx_to_event.append(event)

        return out_dict, np.array(idx_to_event)


class YOLOv5FrameDataset(Dataset):
    def __init__(self, split, cfg: BaseConfig, transform=None, current_idxs: np.ndarray = None, debug=False):
        self.cfg = cfg
        self.split = split
        txt_files = os.listdir(os.path.join(self.cfg.data.data_loc, 'labels', self.split))
        self.annotations = []
        self.txt_files = []

        for txt_file in txt_files:
            with open(os.path.join(self.cfg.data.data_loc, 'labels', self.split, txt_file)) as f:
                self.annotations.extend([f.readlines()])
                self.txt_files.append(txt_file)

        # if self.split == 'test':    # DELETE THIS AS SOON AS FINISHED DEBUGING TEST FUNCTION
        #     self.annotations = self.annotations[:1001]
        #     self.txt_files = self.txt_files[:1001]
        self.shapes = None
        self.labels_for_round = []  # stores all the labels in proper format
        self.global_idxs = np.arange(len(self.annotations))
        self.annotations = np.array(self.annotations, dtype=object)
        self.total_samples = self.annotations.shape[0]
        if current_idxs is not None:
            self.annotations = self.annotations[current_idxs]
            self.global_idxs = self.global_idxs[current_idxs]
        self.transform = transform
        self._cfg = cfg
        self.ignore_iou_thresh = 0.5

        self._debug = True
        self.corrupted = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        class_and_labels = self.annotations[index]
        class_and_labels = list(map(lambda s: s.strip(), class_and_labels))
        image = np.array(Image.open(os.path.join(self.cfg.data.data_loc, 'images', self.split,
                                                 self.txt_files[index][:-3] + 'jpg')).convert("RGB"))

        
        bboxes = np.array([list(map(float, box.split(' '))) for box in class_and_labels]).astype('float')

        classes = bboxes[:, 0]
        bboxes = bboxes[:, 1:]
        bboxes = np.column_stack((bboxes, classes))

        # normalize the bboxes fo yolo format -> normalized [x_center, y_center, width, height]
        im_h, im_w = image.shape[:2]
        bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] / 2) / im_w  # x_center
        bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] / 2) / im_h  # y_center
        bboxes[:, 2] = bboxes[:, 2] / im_w  # width
        bboxes[:, 3] = bboxes[:, 3] / im_h  # height
        bboxes[:, 0:4] = np.clip(bboxes[:, 0:4], a_min=0.0, a_max=0.99)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], 2 * (1 - bboxes[:, 0]))
        bboxes[:, 3] = np.minimum(bboxes[:, 3], 2 * (1 - bboxes[:, 1]))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        bboxes = np.asarray(bboxes)
        classes = bboxes[:, 4]
        bboxes = np.column_stack((classes, bboxes[:, :-1]))
        self.labels_for_round.append(bboxes)
        bboxes = torch.tensor(bboxes)
        labels = torch.zeros((len(bboxes), 6))  # blank, class, x, y, w, h
        h, w = image.shape[:2]
        shapes = (im_h, im_w), ((h / im_h, w / im_w), (0, 0))

        self.shapes = shapes
        labels[:, 1:] = bboxes

        image = image.clone().detach()  # torch.tensor(image)

        paths = os.path.join(self.cfg.data.data_loc, 'images', self.split, self.txt_files[index][:-3] + 'jpg')
        return image, labels, index, self.global_idxs[index], paths, shapes

    def get_event_dict(self):
        out_dict = defaultdict(list)
        idx_to_event = []
        meta_info = pd.read_excel('./visualization/cost/FocalBoxCountEstimatesCost.xlsx')
        event_to_meta = []
        for i in range(self.annotations.shape[0]):
            event = self.txt_files[i][:-14]
            event = event[:-1] if event.endswith('_') else event
            out_dict[event].append(i)
            idx_to_event.append(event)
            # 0 == motion, 1 == box_count_est
            if [event, float(meta_info.loc[meta_info['SequenceID'] == event, 'Motion'].values),
                float(meta_info.loc[meta_info['SequenceID'] == event, 'BoxCountEstimate'].values)] not in event_to_meta:
                event_to_meta.append([event, float(meta_info.loc[meta_info['SequenceID'] == event, 'Motion'].values),
                                      float(meta_info.loc[meta_info['SequenceID'] == event, 'BoxCountEstimate'].values)])
        return out_dict, np.array(idx_to_event), np.asarray(event_to_meta)

    @staticmethod  # Yash added for Yolov5
    def collate_fn(batch):
        im, label, indx, gindx, paths, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), indx, gindx, paths, shapes


if __name__ == '__main__':
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]  # Note these have been rescaled to be between [0, 1]
    IMAGE_SIZE = 832
    transform = A.Compose(
        [
            # A.Resize(IMAGE_SIZE,IMAGE_SIZE),
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )
    import argparse
    import toml
    import os

    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')
    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
