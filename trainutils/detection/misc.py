import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
from Models import build_detection_architecture
from data.loader.seqdatainterface import NUM_CLASSES
from trainutils.detection.nms import non_max_suppression
from trainutils.detection.metrics import bbox_iou_numpy, compute_ap
from trainutils.training.optimization import determine_multilr_milestones
from data import get_yolo_dataset
from trainutils.progresstracking.saver import Saver
from trainutils.progresstracking.summaries import TensorboardSummary
from config import BaseConfig


def get_bbox_images_targets(images: torch.tensor, targets: torch.tensor):
    images = images.clone()
    # targets = targets.clone().detach().cpu().numpy()
    for batch in range(images.shape[0]):
        cur_boxes_scale_batch = targets[batch]

        for box in range(len(cur_boxes_scale_batch)):
            # TODO: Assumes square image!!!!
            # print(cur_boxes_scale_batch[box])
            # print(images.shape)
            if cur_boxes_scale_batch[box].sum() == 0:
                continue
            cls, x, y, width, height = cur_boxes_scale_batch[box]
            top_left_x = max(int((x - width / 2) * images.shape[2]), 0)

            top_left_y = max(int((y - height / 2) * images.shape[3]), 0)
            unnorm_width = int(width * images.shape[3])
            unnorm_height = int(height * images.shape[2])
            right_x = min(images.shape[3] - 1, top_left_x + unnorm_width)
            bottom_y = min(images.shape[2] - 1, top_left_y + unnorm_height)
            r = 1.0
            g = 0
            b = 0
            # R
            images[batch, 0, top_left_y, top_left_x: right_x] = r
            images[batch, 0, bottom_y, top_left_x: right_x] = r
            images[batch, 0, top_left_y: bottom_y, top_left_x] = r
            images[batch, 0, top_left_y: bottom_y, right_x] = r
            # G
            images[batch, 1, top_left_y, top_left_x: right_x] = g
            images[batch, 1, bottom_y, top_left_x: right_x] = g
            images[batch, 1, top_left_y: bottom_y, top_left_x] = g
            images[batch, 1, top_left_y: bottom_y, right_x] = g
            # B
            images[batch, 2, top_left_y, top_left_x: right_x] = b
            images[batch, 2, bottom_y, top_left_x: right_x] = b
            images[batch, 2, top_left_y: bottom_y, top_left_x] = b
            images[batch, 2, top_left_y: bottom_y, right_x] = b
    return images


def get_bbox_images_preds(images: torch.tensor, targets: torch.tensor):
    images = images.clone()
    # targets = targets.clone().detach().cpu().numpy()
    for batch in range(images.shape[0]):
        cur_boxes_scale_batch = targets[batch]
        if cur_boxes_scale_batch is None:
            continue

        for box in range(len(cur_boxes_scale_batch)):
            # TODO: Assumes square image!!!!
            # print(cur_boxes_scale_batch[box])
            top_left_x, top_left_y, right_x, bottom_y, _, _, _ = cur_boxes_scale_batch[box]
            top_left_x, top_left_y, right_x, bottom_y = int(top_left_x), min(int(top_left_y), images.shape[2] - 1), \
                                                        min(int(right_x), images.shape[3] - 1), min(int(bottom_y), images.shape[2] - 1)
            r = 1.0
            g = 0
            b = 0
            # R
            images[batch, 0, top_left_y, top_left_x: right_x] = r
            images[batch, 0, bottom_y, top_left_x: right_x] = r
            images[batch, 0, top_left_y: bottom_y, top_left_x] = r
            images[batch, 0, top_left_y: bottom_y, right_x] = r
            # G
            images[batch, 1, top_left_y, top_left_x: right_x] = g
            images[batch, 1, bottom_y, top_left_x: right_x] = g
            images[batch, 1, top_left_y: bottom_y, top_left_x] = g
            images[batch, 1, top_left_y: bottom_y, right_x] = g
            # B
            images[batch, 2, top_left_y, top_left_x: right_x] = b
            images[batch, 2, bottom_y, top_left_x: right_x] = b
            images[batch, 2, top_left_y: bottom_y, top_left_x] = b
            images[batch, 2, top_left_y: bottom_y, right_x] = b
    return images