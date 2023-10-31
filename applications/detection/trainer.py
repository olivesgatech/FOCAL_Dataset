import os
import tqdm
import torch
import numpy as np
# import torch.nn as nn
from Models import build_detection_architecture
from data.loader.seqdatainterface import NUM_CLASSES
# from trainutils.detection.nms import non_max_suppression
from Models.detection.yolov5.utils.general import non_max_suppression
from trainutils.detection.metrics import bbox_iou_numpy, compute_ap
from trainutils.detection.misc import get_bbox_images_targets, get_bbox_images_preds
# from trainutils.training.optimization import determine_multilr_milestones
from trainutils.detection.confusionmatrix import ConfusionMatrix as ConfMatRyan
from data import get_yolo_dataset
from trainutils.progresstracking.saver import Saver
from trainutils.progresstracking.summaries import TensorboardSummary
from config import BaseConfig
from Models.detection.yolov5.utils.loss import ComputeLoss  # Yash added
import yaml  # yash added
from Models.detection.yolov5.utils.torch_utils import de_parallel, smart_optimizer, torch_distributed_zero_first  # yash added
from Models.detection.yolov5.utils.general import check_amp, scale_boxes, xywh2xyxy, xyxy2xywh, \
    labels_to_class_weights, check_dataset, check_img_size  # yash added
from Models.detection.yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou  # Yash added
from Models.detection.yolov5.val import process_batch
from Models.detection.yolov5.utils.plots import plot_images, output_to_target
from pathlib import Path
from Models.detection.yolov5.utils.callbacks import Callbacks
from Models.detection.yolov5.utils.loggers import Loggers
from Models.detection.yolov5.utils.general import colorstr, methods
from Models.detection.yolov5.utils.autoanchor import check_anchors
from copy import deepcopy
from datetime import datetime

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class DetectionTrainer:
    def __init__(self, cfg: BaseConfig, LOGGER, opt=None):
        self._cfg = cfg
        self._opt = opt
        self._epochs = cfg.detection.epochs
        self._device = cfg.run_configs.gpu_id
        self._eval_training_mapval = False

        # Define Saver
        self._saver = Saver(cfg)
        # self.LOGGER = LOGGER

        # Define Tensorboard Summary
        # self._summary = TensorboardSummary(self._saver.experiment_dir)
        # self._writer = self._summary.create_summary()

        self._callbacks = Callbacks()
        # self._callbacks.run('on_pretrain_routine_start')

        # Hyperparameters
        # hyp = self._cfg.detection.hyp
        # if isinstance(hyp, str):
        #     with open(hyp, errors='ignore') as f:
        #         hyp = yaml.safe_load(f)  # load hyps dict
        #
        # self.LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        # self._hyp = hyp.copy()  # for saving hyps to checkpoints
        #
        # self._data_dict = None

        # if RANK in {-1, 0}:
        #     self._loggers = Loggers(Path(self._cfg.run_configs.ld_folder_name), self._cfg.detection.pretrained, opt, hyp, self.LOGGER)  # loggers instance
        #
        #     # Register actions
        #     for k in methods(self._loggers):
        #         self._callbacks.register_action(k, callback=getattr(self._loggers, k))
        #
        #     # Process custom dataset artifact link
        #     data_dict = self._loggers.remote_dataset
        #
        # with torch_distributed_zero_first(LOCAL_RANK):
        #     data_dict = data_dict or check_dataset(self._cfg.detection.data)  # check if None
        #
        # self._data_dict = data_dict
        #
        # self._nc = int(data_dict['nc'])  # number of classes
        # self._names = data_dict['names']  # class names

        # confusion matrix
        # self._conf_matrix_ryan = ConfMatRyan(num_classes=NUM_CLASSES, CONF_THRESHOLD=cfg.detection.conf_threshold,
        #                                     IOU_THRESHOLD=cfg.detection.map_iou_threshold)
        # self._conf_matrix = ConfusionMatrix(nc=self._cfg.data.num_classes)
        # self._seen = 0

        # TODO: make this generalizable
        self._loaders = get_yolo_dataset(cfg)

        # self._model = build_detection_architecture(cfg)

        # check_anchors(self._loaders.train_loader.dataset, model=self._model, thr=self._hyp['anchor_t'],
        #               imgsz=self._cfg.detection.img_size)  # run AutoAnchor

        # Yash added
        # self._amp = check_amp(self._model)
        #
        # # No freezing
        # for k, v in self._model.named_parameters():
        #     v.requires_grad = True  # train all layers

        # Image size
        # self._gs = max(int(self._model.stride.max()), 32)  # grid size (max stride)
        # self._imgsz = check_img_size(self._cfg.detection.img_size, self._gs, floor=self._gs * 2)

        # self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp)
        # self._last_opt_step = -1
        #
        # self._nbs = 64
        # self._accumulate = max(round(self._nbs / self._cfg.detection.batch_size), 1)  # accumulate loss before optimizing
        # self._hyp['weight_decay'] *= self._cfg.detection.batch_size * self._accumulate / self._nbs  # scale weight_decay
        # self._optimizer = smart_optimizer(self._model, opt.optimizer, self._hyp['lr0'], self._hyp['momentum'], self._hyp['weight_decay'])
        # self._lf = lambda x: (1 - x / self._cfg.active_learning.max_epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        # self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=self._lf)
        #
        # nl = de_parallel(self._model).model[-1].nl  # number of detection layers (to scale hyps)
        # self._hyp['box'] *= 3 / nl  # scale to layers
        # self._hyp['cls'] *= self._cfg.data.num_classes / 80 * 3 / nl  # scale to classes and layers
        # self._hyp['obj'] *= (self._cfg.detection.img_size / 640) ** 2 * 3 / nl  # scale to image size and layers
        # self._model.nc = self._cfg.data.num_classes  # attach number of classes to model
        # self._model.hyp = self._hyp  # yash added

        # self._best_fitness, self._start_epoch = 0.0, 0

        # Using cuda
        # if self._cfg.run_configs.cuda:
        #     # use multiple GPUs if available
        #     self._model = torch.nn.DataParallel(self._model, device_ids=[self._cfg.run_configs.gpu_id])
        #     # self._model.hyp = self._hyp
        #     # self._model = smart_DDP(model)
        # else:
        #     self._device = torch.device('cpu')

        # if self._cfg.run_configs.resume != 'none':
        #     resume_file = self._cfg.run_configs.resume
        #     # we have a checkpoint
        #     if not os.path.isfile(resume_file):
        #         raise RuntimeError("=> no checkpoint found at '{}'".format(resume_file))
        #     # load checkpoint
        #     checkpoint = torch.load(resume_file)
        #     # minor difference if working with cuda
        #     if self._cfg.run_configs.cuda:
        #         self._model.module.load_state_dict(checkpoint['state_dict'])
        #         # self._model.load_state_dict(checkpoint['state_dict'])
        #     else:
        #         self._model.load_state_dict(checkpoint['state_dict'])
        #     # self._optimizer.load_state_dict(checkpoint['optimizer'])
        #     print("=> loaded checkpoint '{}' )".format(resume_file))
        #
        #     # If we don't do this then it will just have learning rate of old checkpoint
        #     # and it will lead to many hours of debugging \:
        #     for param_group in self._optimizer.param_groups:
        #         param_group["lr"] = self._cfg.detection.optimization.lr

        # init best object acc
        # self._best_map = 0.0

    def training(self, epoch, save_checkpoint=False, track_summaries=False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.train()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)

        # init statistics parameters
        train_loss = 0.0

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)
                target = target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            loss = self._model(image, target)

            # perform backpropagation
            loss.backward()

            # update params with gradients
            self._optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            if track_summaries:
                with torch.no_grad():
                    output = self._model(image)
                self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
                if i % 10 == 0:
                    track_predictions = False
                    if epoch > 0 and epoch % 15 == 0 and not track_predictions:
                        track_predictions = True
                    else:
                        track_predictions = False
                    self._summary.visualize_image(self._writer, image, target, output,
                                                  self._cfg.detection.iou_threshold,
                                                  self._cfg.detection.conf_threshold,
                                                  self._device,
                                                  track_predictions,
                                                  i + num_img_tr * epoch)

        # Update optimizer step
        if self._cfg.detection.optimization.scheduler != 'none':
            self._scheduler.step(epoch)

        # calculate accuracy
        self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.detection.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # save checkpoint
        if not self._cfg.run_configs.val:
            self._saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            })

        return

    def training_yolov5(self, epoch, save_checkpoint=False, track_summaries=False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._callbacks.run('on_train_epoch_start')
        self._model.train()
        # initializes cool bar for visualization
        nb = len(self._loaders.train_loader)
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)
        self._scheduler.last_epoch = self._start_epoch - 1

        last_opt_step = -1
        maps = np.zeros(self._nc)  # mAP per class

        # init statistics parameters
        train_loss = 0.0

        nw = max(round(self._hyp['warmup_epochs'] * nb), 100)  # self._hyp['warmup_epochs']  # m

        self._model.names = self._names
        self._compute_loss = ComputeLoss(self._model)  # Yash added

        self._optimizer.zero_grad()

        mloss = torch.zeros(3, device=self._device)  # mean losses
        # self.LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))

        # iterate over all samples in each batch i
        for i, (image, target, idxs, _, paths, _) in enumerate(tbar):
            self._callbacks.run('on_train_batch_start')
            # attach class weights
            self._model.class_weights = labels_to_class_weights(self._loaders.train_loader.dataset.labels_for_round,
                                                                self._model.module.nc).to(self._device) * self._model.module.nc

            ni = i + nb * epoch  # number integrated batches (since train start)

            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                self._accumulate = max(1, np.interp(ni, xi, [1, self._nbs / self._cfg.detection.batch_size]).round())
                for j, x in enumerate(self._optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [self._hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self._lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self._hyp['warmup_momentum'], self._hyp['momentum']])

            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)
                target = target.to(self._device)

            # convert image to suitable dims
            image = image.float()

            # Yash added this to work for Yolov5
            with torch.cuda.amp.autocast(self._amp):
                output = self._model(image)  # yash added
                loss, loss_items = self._compute_loss(output, target)  # yash added
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # perform backpropagation
            self._scaler.scale(loss).backward()

            # update params with gradients Yash modified
            if ni - self._last_opt_step >= self._accumulate:
                self._scaler.unscale_(self._optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)  # clip gradients
                self._scaler.step(self._optimizer)  # optimizer.step
                # old_scaler = self._scaler.get_scale()
                self._scaler.update()
                # new_scaler = self._scaler.get_scale()
                # if new_scaler < old_scaler:
                #     self._scheduler.step(epoch)
                self._optimizer.zero_grad()
                self._last_opt_step = ni

            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                #                      (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                self._callbacks.run('on_train_batch_end', self._model, ni, image, target, paths, list(mloss))

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            if track_summaries:
                with torch.no_grad():
                    output = self._model(image)
                self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
                if i % 10 == 0:
                    track_predictions = False
                    if epoch > 0 and epoch % 15 == 0 and not track_predictions:
                        track_predictions = True
                    else:
                        track_predictions = False
                    self._summary.visualize_image(self._writer, image, target, output,
                                                  self._cfg.detection.iou_threshold,
                                                  self._cfg.detection.conf_threshold,
                                                  self._device,
                                                  track_predictions,
                                                  i + num_img_tr * epoch)

        # if self._cfg.detection.optimization.scheduler != 'none':
        #     self._scheduler.step(epoch)
        lr = [x['lr'] for x in self._optimizer.param_groups]
        self._scheduler.step()

        self._callbacks.run('on_train_epoch_end', epoch=epoch)

        # calculate accuracy
        self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.detection.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # save checkpoint
        self.checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'best_fitness': self._best_fitness,
            'model': deepcopy(de_parallel(self._model)).half(),
            # 'ema': None
            # 'updates': None
            'wandb_id': self._loggers.wandb.wandb_run.id if self._loggers.wandb else None,
            'opt': vars(self._opt),
            'date': datetime.now().isoformat()
        }
        torch.save(self.checkpoint, self._cfg.run_configs.output_folder_name + 'last.pt')
        return lr, mloss

    def testing_yolov5(self):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization

        conf_matr = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))

        all_detections = []
        all_annotations = []

        names = self._model.names if hasattr(self._model, 'names') else self._model.module.names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))

        # s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        # tbar = tqdm.tqdm(self._loaders.test_loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        tbar = tqdm.tqdm(self._loaders.test_loader)
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss = torch.zeros(3, device=self._device)
        jdict, stats, ap, ap_class = [], [], [], []

        self._callbacks.run('on_val_start')

        self._compute_loss = ComputeLoss(self._model)  # Yash added

        # iterate over all samples in each batch i
        for i, (image, target, idxs, _, paths, shapes) in enumerate(tbar):

            self._callbacks.run('on_val_batch_start')

            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)
                target = target.to(self._device)

            # convert image to suitable dims
            image = image.float()  # uint8 to fp16/32
            image /= 255
            nb, _, height, width = image.shape
            iouv = torch.linspace(0.5, 0.95, 10, device=self._device)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()

            # computes output of our model
            with torch.no_grad():
                preds, train_out = self._model(image) if self._compute_loss else (self._model(image, augment=False), None)
                loss += self._compute_loss(train_out, target)[1]  # box, obj, cls
                target[:, 2:] *= torch.tensor((width, height, width, height), device=self._device)  # to pixels
                lb = [target[target[:, 0] == i, 1:] for i in range(nb)] if False else []
                preds = non_max_suppression(preds, self._cfg.detection.conf_threshold,
                                              self._cfg.detection.nms_iou_threshold, labels=lb,
                                              multi_label=True, max_det=300)

            if self._cfg.run_configs.save_test_imgs:
                img_targets = get_bbox_images_targets(image, target)
                img_preds = get_bbox_images_preds(image, preds)
                self._saver.save_test_imgs(img_preds.permute(0, 2, 3, 1).cpu().numpy(),
                                           img_targets.permute(0, 2, 3, 1).cpu().numpy())

            # Metrics
            for si, pred in enumerate(preds):
                labels = target[target[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), [2048, 2448]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=self._device)  # init
                self._seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=self._device), labels[:, 0]))
                        self._conf_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue

                # Predictions
                predn = pred.clone()
                scale_boxes(image[si].shape[1:], predn[:, :4], shape)  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(image[si].shape[1:], tbox, shape)  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    self._conf_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
                self._callbacks.run('on_val_image_end', pred, predn, path, names, image[si])

            # Plot images
            if i < 3:
                plot_images(image, target, paths,
                            os.path.join(self._cfg.run_configs.ld_folder_name, f'test_batch{i}_labels.jpg'), names)  # labels
                plot_images(image, output_to_target(preds), paths,
                            os.path.join(self._cfg.run_configs.ld_folder_name, f'test_batch{i}_pred.jpg'), names)  # pred
            self._callbacks.run('on_val_batch_end', i, image, target, paths, shapes, preds)

            for output, annotations in zip(preds, target.cpu()):

                all_detections.append([np.array([]) for _ in range(NUM_CLASSES)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()  # confidence
                    pred_labels = output[:, -1].cpu().numpy()

                    # order output as x1, y1, x2, y2, conf, class for conf matrix
                    # print(f'bboxes: {pred_boxes.shape} scores: {scores.shape} labels: {pred_labels.shape}')
                    det_batch = np.concatenate((pred_boxes, scores[..., np.newaxis], pred_labels[..., np.newaxis]),
                                               axis=1)

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(NUM_CLASSES):
                        all_detections[-1][label] = pred_boxes[pred_labels == label]
                else:
                    # no detections so send nothing to conf matrix
                    det_batch = None

                all_annotations.append([np.array([]) for _ in range(NUM_CLASSES)])
                # if any(annotations[:, -1] > 0):
                if annotations[-1] > 0:
                    annotation_labels = annotations[1].cpu().numpy()
                    _annotation_boxes = annotations[2:]

                    # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[0] = _annotation_boxes[0] - _annotation_boxes[2] / 2
                    annotation_boxes[1] = _annotation_boxes[1] - _annotation_boxes[3] / 2
                    annotation_boxes[2] = _annotation_boxes[0] + _annotation_boxes[2] / 2
                    annotation_boxes[3] = _annotation_boxes[1] + _annotation_boxes[3] / 2
                    annotation_boxes *= self._cfg.detection.img_size

                    # update conf matrix TODO: Conf matrix only considers images with annotations!!!
                    ann_batch = np.concatenate((annotation_labels[..., np.newaxis], annotation_boxes), axis=0)
                    self._conf_matrix_ryan.process_batch_yolov5(det_batch, ann_batch)

                    for label in range(NUM_CLASSES):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True,
                                                          save_dir=self._cfg.run_configs.ld_folder_name, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            print('yolo map50: ' + str(map50))

        nt = np.bincount(stats[3].astype(int), minlength=NUM_CLASSES)  # number of targets per class
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        self.LOGGER.info(pf % ('all', self._seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            self.LOGGER.warning(f'WARNING ⚠️ no labels found in test set, can not compute metrics without labels')

        # Print results per class
        for i, c in enumerate(ap_class):
            self.LOGGER.info(pf % (names[c], self._seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        self._conf_matrix.plot(save_dir=self._cfg.run_configs.ld_folder_name, names=list(names.values()))
        self._callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, self._conf_matrix)

        average_precisions = {}
        for label in range(NUM_CLASSES):
            true_positives = []
            scores = []
            num_annotations = 0

            for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]

                num_annotations += annotations.shape[0]
                detected_annotations = []

                for *bbox, score in detections:
                    scores.append(score)

                    if annotations.shape[0] == 0:
                        # no annotation for that class so everything is wrong
                        true_positives.append(0)
                        conf_matr[-1, label] += 1
                        continue

                    overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self._cfg.detection.map_iou_threshold and assigned_annotation not in detected_annotations:
                        true_positives.append(1)
                        conf_matr[label, label] += 1
                        detected_annotations.append(assigned_annotation)
                    else:
                        if max_overlap < self._cfg.detection.map_iou_threshold:
                            conf_matr[-1, label] += 1
                        true_positives.append(0)

                conf_matr[label, -1] = num_annotations - len(detected_annotations)

            # no annotations -> AP for this class is 0
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            true_positives = np.array(true_positives)
            false_positives = np.ones_like(true_positives) - true_positives
            # sort by score
            indices = np.argsort(-np.array(scores))
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        s = f"\n{len(list(Path(self._cfg.run_configs.ld_folder_name).glob('labels/*.txt')))} labels saved to {Path(self._cfg.run_configs.ld_folder_name + '/labels')}"
        self.LOGGER.info(f"Results saved to {colorstr('bold', self._cfg.run_configs.ld_folder_name)}{s}")
        maps = np.zeros(self._nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        print("Average Precisions:")
        test_output = {}
        for c, ap in average_precisions.items():
            print(f"Class '{c}' - AP: {ap}")
            test_output[c] = ap
        mAP = np.mean(list(average_precisions.values()))
        print(f"mAP: {mAP}")
        print('Confusion Matrix:')
        self._conf_matrix_ryan.print_matrix()
        print('Confusion Matrix Self Calculated')
        for i in range(NUM_CLASSES + 1):
            print(np.array2string(conf_matr[i, :]))

        test_output['mAP'] = mAP
        return test_output, (mp, mr, map50, map, *(loss.cpu() / len(self._loaders.test_loader)).tolist()), maps

    def testing(self, epoch):
        from trainutils.detection.nms import non_max_suppression

        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.test_loader)
        num_img_tr = len(self._loaders.test_loader)
        conf_matr = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))

        all_detections = []
        all_annotations = []
        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)
                target = target.to(self._device)

            # convert image to suitable dims
            image = image.float()
            # computes output of our model
            with torch.no_grad():
                outputs = self._model(image)
                outputs = non_max_suppression(outputs, NUM_CLASSES, self._cfg.detection.conf_threshold,
                                              self._cfg.detection.nms_iou_threshold)

            if self._cfg.run_configs.save_test_imgs:
                img_targets = get_bbox_images_targets(image, target)
                img_preds = get_bbox_images_preds(image, outputs)
                self._saver.save_test_imgs(img_preds.permute(0, 2, 3, 1).cpu().numpy(),
                                           img_targets.permute(0, 2, 3, 1).cpu().numpy())
            for output, annotations in zip(outputs, target.cpu()):

                all_detections.append([np.array([]) for _ in range(NUM_CLASSES)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    # order output as x1, y1, x2, y2, conf, class for conf matrix
                    # print(f'bboxes: {pred_boxes.shape} scores: {scores.shape} labels: {pred_labels.shape}')
                    det_batch = np.concatenate((pred_boxes, scores[..., np.newaxis], pred_labels[..., np.newaxis]),
                                               axis=1)

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(NUM_CLASSES):
                        all_detections[-1][label] = pred_boxes[pred_labels == label]
                else:
                    # no detections so send nothing to conf matrix
                    det_batch = None

                all_annotations.append([np.array([]) for _ in range(NUM_CLASSES)])
                if any(annotations[:, -1] > 0):

                    annotation_labels = annotations[annotations[:, -1] > 0, 0].cpu().numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                    # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= self._cfg.detection.img_size

                    # update conf matrix TODO: Conf matrix only considers images with annotations!!!
                    ann_batch = np.concatenate((annotation_labels[..., np.newaxis], annotation_boxes), axis=1)
                    self._conf_matrix.process_batch(det_batch, ann_batch)

                    for label in range(NUM_CLASSES):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

        average_precisions = {}
        for label in range(NUM_CLASSES):
            true_positives = []
            scores = []
            num_annotations = 0

            for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]

                num_annotations += annotations.shape[0]
                detected_annotations = []

                for *bbox, score in detections:
                    scores.append(score)

                    if annotations.shape[0] == 0:
                        # no annotation for that class so everything is wrong
                        true_positives.append(0)
                        conf_matr[-1, label] += 1
                        continue

                    overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self._cfg.detection.map_iou_threshold and assigned_annotation not in detected_annotations:
                        true_positives.append(1)
                        conf_matr[label, label] += 1
                        detected_annotations.append(assigned_annotation)
                    else:
                        if max_overlap < self._cfg.detection.map_iou_threshold:
                            conf_matr[-1, label] += 1
                        true_positives.append(0)

                conf_matr[label, -1] = num_annotations - len(detected_annotations)

            # no annotations -> AP for this class is 0
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            true_positives = np.array(true_positives)
            false_positives = np.ones_like(true_positives) - true_positives
            # sort by score
            indices = np.argsort(-np.array(scores))
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        print("Average Precisions:")
        test_output = {}
        for c, ap in average_precisions.items():
            print(f"Class '{c}' - AP: {ap}")
            test_output[c] = ap
        mAP = np.mean(list(average_precisions.values()))
        print(f"mAP: {mAP}")
        print('Confusion Matrix:')
        self._conf_matrix.print_matrix()
        print('Confusion Matrix Self Calculated')
        for i in range(NUM_CLASSES + 1):
            print(' '.join(map(str, conf_matr[i])))

        test_output['mAP'] = mAP
        return test_output

    def get_loaders(self, mode: str):
        if mode == 'test' or mode == 'train':
            return self._loaders
        else:
            raise Exception('Test mode not implemented yet')

    def save_cache(self):
        self._loaders.save_cache()