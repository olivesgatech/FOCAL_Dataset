import collections
import time
import numpy as np
import torch
import tqdm
from data import get_yolo_dataset
from applications.detection.trainer import DetectionTrainer
from data.loader.seqdatainterface import NUM_CLASSES
from trainutils.detection.nms import non_max_suppression
from config import BaseConfig


def append_if_not_none(arr: np.array, app: np.ndarray):
    if arr is not None:
        arr = np.append(arr, app)
    else:
        arr = app

    return arr


class ALDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg: BaseConfig, LOGGER, opt=None):
        super(ALDetectionTrainer, self).__init__(cfg, LOGGER, opt)

        self._train_pool = np.arange(self._loaders.data_config.train_len)
        self.n_pool = self._loaders.data_config.train_len
        self._unlabeled_loader = None
        self._unlabeled_statistics = None
        self._softmax = torch.nn.Softmax(dim=1)
        self._model = None
        # init frame switches
        self.switches = collections.defaultdict(int)
        self._prev_num_obj = {}

    def update_seed(self, seed: int):
        self._saver.seed = seed

    def update_loader(self, idxs: np.ndarray, unused_idxs: np.array):
        self._loaders = get_yolo_dataset(self._cfg, idxs=idxs)
        self._unlabeled_loader = get_yolo_dataset(self._cfg, idxs=unused_idxs)

    def switch_statistics(self):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader, desc='Testing unlabeled samples')
        num_img_tr = len(self._unlabeled_loader.train_loader)

        # init statistics parameters
        train_loss = 0.0

        # init output dict
        total_idxs = None
        switches = []

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs, global_idxs = sample['data'], sample['label'], sample['idx'], sample['global_idx']
            bs = image.shape[0]
            global_idxs = global_idxs.cpu().numpy()
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)

            # convert image to suitable dims
            image = image.float()
            # computes output of our model
            t1 = time.time()
            with torch.no_grad():
                output = self._model(image)
                # output = non_max_suppression(outputs, NUM_CLASSES, self._cfg.detection.conf_threshold,
                #                              self._cfg.detection.nms_iou_threshold)
            t2 = time.time()
            for k in range(output.shape[0]):
                # pseudo nms
                conf_mask = (output[k, :, 4] >= self._cfg.detection.conf_threshold).squeeze()
                num_obj = torch.sum(conf_mask).item()

                if global_idxs[k] not in self._prev_num_obj:
                    self.switches[global_idxs[k]] = 0
                else:
                    self.switches[global_idxs[k]] += abs(num_obj - self._prev_num_obj[global_idxs[k]])

                self._prev_num_obj[global_idxs[k]] = num_obj
                total_idxs = append_if_not_none(total_idxs, global_idxs[k])
                switches.append(self.switches[global_idxs[k]])
            t3 = time.time()

            m_time = t2 - t1
            proc_time = t3 - t2
            tbar.set_description(f'Model: {m_time} Proc.: {proc_time}')

        # calculate accuracy
        stats = {'idxs': total_idxs, 'switches': np.array(switches)}
        print('Loss: %.3f' % train_loss)
        return stats

    def unlabeled_statistics(self,model):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        self._model = model
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader, desc='Testing unlabeled samples')
        num_img_tr = len(self._unlabeled_loader.train_loader)

        # init statistics parameters
        train_loss = 0.0

        # init output dict
        total_idxs = None
        margins = None
        entropies = None
        lconfs = None
        softmax = torch.nn.Softmax()

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            
            #image, target, idxs, global_idxs = sample['data'], sample['label'], sample['idx'], sample['global_idx']
            image,target,idxs,global_idxs,paths,shapes = sample
            #print(idxs)
            #print(global_idxs)
            bs = image.shape[0]
            #global_idxs = global_idxs[0]
            #print(global_idxs)
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image = image.to(self._device)
                target = target.to(self._device)

            # convert image to suitable dims
            image = image.float()
            # computes output of our model
            with torch.no_grad():
                outputs = self._model(image)
                output = non_max_suppression(outputs, NUM_CLASSES, self._cfg.detection.conf_threshold,
                                             self._cfg.detection.nms_iou_threshold)
            for k in range(len(output)):
                if output[k] is not None:

                    obj_scores = output[k][:, 4]
                    # normalize object scores between [0, 1]
                    if obj_scores.shape[0] > 1:
                        obj_scores = softmax(obj_scores)

                        total_objscores, _ = torch.sort(obj_scores)
                        lconf_scores = total_objscores[0]
                        margin_scores = total_objscores[1] - total_objscores[0]

                        logobj_scores = torch.log2(total_objscores)
                        mult = logobj_scores * total_objscores
                        entropy = -1 * torch.sum(mult)
                    else:
                        # handle instances where only 1 element is present.
                        total_objscores = obj_scores
                        lconf_scores = total_objscores[0]
                        margin_scores = total_objscores[0]
                        logobj_scores = torch.log2(total_objscores)
                        entropy = -1 * logobj_scores * total_objscores

                    entropies = append_if_not_none(entropies, entropy.cpu().numpy())
                    margins = append_if_not_none(margins, margin_scores.cpu().numpy())
                    lconfs = append_if_not_none(lconfs, lconf_scores.cpu().numpy())
                else:
                    # no detection in image. Put lowest values for image
                    entropies = append_if_not_none(entropies, np.array([0.0]))
                    margins = append_if_not_none(margins, np.array([1.0]))
                    lconfs = append_if_not_none(lconfs, np.array([1.0]))
                total_idxs = append_if_not_none(total_idxs, global_idxs[k])

        # calculate accuracy
        stats = {'idxs': total_idxs, 'lconf': lconfs, 'margin': margins, 'entropy': entropies}
        print('Loss: %.3f' % train_loss)
        return stats
