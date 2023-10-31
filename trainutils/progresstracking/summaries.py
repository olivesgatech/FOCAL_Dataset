import os
import torch
from torchvision.utils import make_grid
from data.utils.boxutils import cells_to_bboxes, non_max_suppression
from Models.detection.yolov3DEP.yolocfg import ANCHORS
from tensorboardX import SummaryWriter
import numpy as np


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer: SummaryWriter,
                        image: torch.tensor,
                        target: tuple,
                        output: list,
                        nms_threshold: float,
                        conf_threshold: float,
                        device: torch.device,
                        track_prediction: bool,
                        global_step: int):
        rel_batch = image[:3]
        rel_targets = target[:3]


        # get target bboxes
        bbox_targets = rel_targets

        drawn_bboxes_targets = self.draw_bboxes(rel_batch, rel_targets)
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(drawn_bboxes_targets.cpu().data, 3, normalize=False, range=(0, 255))
        writer.add_image('GT Label', grid_image, global_step)
        rel_outputs = {}
        rel_outputs[0] = output[0][:3]
        rel_outputs[1] = output[1][:3]
        rel_outputs[2] = output[2][:3]

        if track_prediction:
            # get image bboxes
            bbox_img = [[] for _ in range(rel_batch.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = rel_outputs[i].shape
                anchor = torch.tensor([*ANCHORS[i]]).to(device) * S
                boxes_scale_i = cells_to_bboxes(rel_outputs[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bbox_img[idx] += box
            bbox_img_nms = [[] for _ in range(rel_batch.shape[0])]
            for i in range(batch_size):
                img_nms = non_max_suppression(bbox_img[i],
                                              iou_threshold=nms_threshold,
                                              threshold=conf_threshold,
                                              box_format="midpoint")
                bbox_img_nms[i].extend(img_nms)
            drawn_bboxes_pred = self.draw_bboxes(rel_batch, bbox_img_nms)
            grid_image = make_grid(drawn_bboxes_pred.cpu().data, 3, normalize=False, range=(0, 255))
            writer.add_image('Predicted', grid_image, global_step)

    def draw_bboxes(self, images: torch.tensor, targets: torch.tensor):
        images = images.clone()
        # targets = targets.clone().detach().cpu().numpy()
        for batch in range(images.shape[0]):

            for box in range(targets.shape[0]):
                # TODO: Assumes square image!!!!
                # print(cur_boxes_scale_batch[box])
                if torch.sum(targets[box, :]) == 0:
                    continue
                class_label, x, y, width, height = targets[batch, box, :]
                top_left_x = max(int((x - width / 2)*images.shape[2]), 0)

                top_left_y = max(int((y - height / 2)*images.shape[3]), 0)
                unnorm_width = int(width*images.shape[3])
                unnorm_height = int(height*images.shape[2])
                right_x = min(images.shape[3] - 1, top_left_x + unnorm_width)
                bottom_y = min(images.shape[2] - 1, top_left_y + unnorm_height)
                #print(images.shape)
                #print(cur_boxes_scale_batch[box])
                #print(top_left_x)
                #print(top_left_y)
                #print(unnorm_height)
                #print(unnorm_width)
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
