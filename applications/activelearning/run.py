import random

import numpy as np
import toml
import os
import sys
import shutil
import argparse
import torch
import pandas as pd
from data import get_yolo_dataset
from applications.activelearning.qustrategies import *
from applications.activelearning.trainer import ALDetectionTrainer
from applications.activelearning.alutils.sequtils import seq_to_idxs
from pathlib import Path
from Models.detection.yolov5.utils.general import LOGGER, colorstr
from Models.detection.yolov5 import val, train
from Models.detection.yolov5.utils.general import check_yaml, print_args
from Models.detection.yolov5.models.experimental import attempt_load
from Models.detection.yolov5.utils.dataloaders import create_dataloader
import collections

LOGGER.parent.removeHandler(LOGGER.parent.handlers[1])

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def make_dir(path):
    if not os.path.exists(os.path.expanduser(path)):
        os.mkdir(os.path.expanduser(path))


def main(cfg: BaseConfig, opt=None, opt_val=None):
    delattr(opt, 'config')

    np.random.seed(cfg.active_learning.init_seed)

    if not os.path.exists(os.path.expanduser(cfg.run_configs.ld_folder_name)):
        os.makedirs(os.path.expanduser(cfg.run_configs.ld_folder_name))

    # get all relevant statistics for the dataset
    train_configs = get_yolo_dataset(cfg)
    event_dict, idx_to_event, meta_dict = train_configs.train_loader.dataset.get_event_dict()
    event_list = list(event_dict.keys())  # list of sequences
    event_list = [event.split('-')[-1] for event in event_list]
    np.save(cfg.run_configs.ld_folder_name + 'event_list.npy', np.array(event_list))
    if cfg.active_learning.sampling_type == 'frame':
        n_pool = len(os.listdir(cfg.data.data_loc))

    else:
        n_pool = cfg.data.total_sequences
    print(f'Number of Samples: {n_pool}')
    nquery = cfg.active_learning.n_query
    nstart = cfg.active_learning.n_start
    nend = cfg.active_learning.n_end
    minimum = None
    if cfg.active_learning.sampling_type == 'sequence':
        if cfg.active_learning.strategy == 'leastframe':
            sorted_events = collections.OrderedDict(sorted(event_dict.items(), key=lambda x: len(x[1]), reverse=False))
            start_seqs = np.array(list(sorted_events.keys())[:nstart])
            start_idxs = []
            for seq in start_seqs:
                start_idxs.append(event_list.index(seq.split('-')[-1]))
        elif cfg.active_learning.strategy == 'mostframe':
            sorted_events = collections.OrderedDict(sorted(event_dict.items(), key=lambda x: len(x[1]), reverse=True))
            start_seqs = np.array(list(sorted_events.keys())[:nstart])
            start_idxs = []
            for seq in start_seqs:
                start_idxs.append(event_list.index(seq.split('-')[-1]))
        elif cfg.active_learning.strategy == 'minmaxmotion':
            sorted_motion = meta_dict[:, 1].argsort()  # sort motion in increasing order
            # start with min motion at round 0
            start_seqs = meta_dict[sorted_motion[:int(nstart)]]
            # set minimum to false for next round
            minimum = False
            # if nstart % 2 == 0:
            #     start_seqs = np.vstack((meta_dict[sorted_motion[:int(nstart/2)]],
            #                             meta_dict[sorted_motion[-int(nstart/2):]]))
            # else:
            #     start_seqs = np.vstack((meta_dict[sorted_motion[:int(nstart/2) + 1]],
            #                             meta_dict[sorted_motion[-(int(nstart/2) + 1):]]))
            start_idxs = []
            for seq in start_seqs:
                start_idxs.append(event_list.index(seq[0].split('-')[-1]))
        elif cfg.active_learning.strategy == 'minmotion':
            sorted_motion = meta_dict[:, 1].argsort()  # sort motion in increasing order
            start_seqs = meta_dict[sorted_motion[:int(nstart)]]
            start_idxs = []
            for seq in start_seqs:
                start_idxs.append(event_list.index(seq[0].split('-')[-1]))
        elif cfg.active_learning.strategy == 'minboxes':
            sorted_motion = meta_dict[:, 2].argsort()  # sort box estimates in increasing order
            start_seqs = meta_dict[sorted_motion[:int(nstart)]]
            start_idxs = []
            for seq in start_seqs:
                start_idxs.append(event_list.index(seq[0].split('-')[-1]))
        else:
            start_idxs = np.arange(n_pool)[np.random.permutation(n_pool)][:nstart]
    else:
        start_idxs = np.arange(n_pool)[np.random.permutation(n_pool)][:nstart]
        

    if nend < n_pool:
        nrounds = int((nend - nstart) / nquery)
        print('Rounds: %d' % nrounds)
    else:
        nrounds = int((n_pool - nstart) / nquery) + 1
        print('Number of end samples too large! Using total number of samples instead. Rounds: %d Total Samples: %d' %
              (nrounds, n_pool))

    for i in range(cfg.active_learning.start_seed, cfg.active_learning.end_seed):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)
        opt_train.seed = i
        if cfg.active_learning.sampling_type == 'frame':
            sampler = get_frame_sampler(cfg=cfg, n_pool=n_pool, start_idxs=start_idxs)
        elif cfg.active_learning.sampling_type == 'sequence':
            addons = {'total_frames': len(os.listdir(os.path.join(cfg.data.data_loc, 'images', 'train')))} #train_configs.data_config.train_len}
            sampler = get_sequence_sampler(cfg=cfg, n_pool=n_pool, start_idxs=start_idxs, event_dict=event_dict,
                                           addons=addons, meta_dict=meta_dict, event_list=event_list, minimum=minimum)
        else:
            raise Exception('Only Frame or Sequence AL supported!')

        all_results = pd.DataFrame([])  # , columns=['Precision', 'Recall', 'mAP@.5', 'mAP@.5-.95', 'test_loss(box)',
                                                         # 'test_loss(obj)', 'test_loss(cls)']

        for round in range(nrounds):
            trainer = ALDetectionTrainer(cfg, LOGGER, opt)

            trainer.update_seed(i)
            if cfg.active_learning.sampling_type == 'frame':
                cur_idxs = sampler.idx_current.astype(int)
                unlabeled = np.squeeze(np.argwhere(sampler.total_pool == 0))
            else:
                cur_idxs = seq_to_idxs(event_dict, sampler.idx_current.astype(int))  # convert current sequences to indices
                unlabeled = seq_to_idxs(event_dict, np.squeeze(np.argwhere(sampler.total_pool == 0)))
            trainer.update_loader(cur_idxs, unlabeled)

            epoch = 1

            if not os.path.exists(os.path.expanduser(cfg.run_configs.ld_folder_name + 'seed_' + str(i))):
                os.makedirs(os.path.expanduser(cfg.run_configs.ld_folder_name + 'seed_' + str(i)))
            round_folder = cfg.run_configs.ld_folder_name + 'seed_' + str(i) + '/round' + str(round)
            make_dir(round_folder)
            target_folder = round_folder + '/seed' + str(i) + '/'
            make_dir(target_folder)

            if round == 0:
                if cfg.active_learning.sampling_type == 'sequence':
                    new_events = np.array(list(event_dict.keys()))[start_idxs]
                    np.save(target_folder + 'new_events_round0.npy', new_events)
                elif cfg.active_learning.sampling_type == 'frame':
                    new_events = idx_to_event[start_idxs]  # convert seqs to indices
                    np.save(target_folder + 'new_events_round0.npy', new_events)

            print('Round: %d' % round)
            print('Seed: %d' % i)
            #print(sampler.idx_current)
            #print(len(sampler.idx_current))
            #return 0
            # Trains for the set number of epochs and validates on val set after each epoch
            if cfg.active_learning.sampling_type == 'frame':
                _, data_dict, gs, callbks, compute_loss,model = train.train(opt_train.hyp, opt_train,
                                                                      torch.device('cuda'),
                                                                      trainer._callbacks,sampler.idx_current, ROUND=round, seed=i)
            else:
                _, data_dict, gs, callbks, compute_loss,model = train.train(opt_train.hyp, opt_train,
                                                                      torch.device('cuda'),
                                                                      trainer._callbacks,
                                                                      sampler.idx_current, round, seed=i)
            print('Round: %d' % round)
            print('Seed: %d' % i)

            # perform sampling action
            sampler.action(trainer, epoch)

            new_idxs = sampler.query(model,nquery, trainer)  # outputs sequences
            if cfg.active_learning.sampling_type == 'sequence':
                new_events = np.array(list(event_dict.keys()))[new_idxs]
                print(new_events)
                np.save(target_folder + 'new_events.npy', new_events)
            elif cfg.active_learning.sampling_type == 'frame':
                new_events = idx_to_event[new_idxs]  # convert seqs to indices
                print(new_events)
                np.save(target_folder + 'new_events.npy', new_events)

            sampler.save_data(target_folder)
            sampler.update(new_idxs)

            # Evaluates best model on test set after training is complete
            test_loader = create_dataloader(data_dict['test'],
                                           opt_val.imgsz,
                                           opt_val.batch_size // WORLD_SIZE * 2,
                                           gs,
                                           False,
                                           hyp=opt_train.hyp,
                                           cache=None if opt_train.noval else opt.cache,
                                           rect=True,
                                           rank=-1,
                                           workers=opt_train.workers * 2,
                                           pad=0.5,
                                           prefix=colorstr('test: '))[0]
            results, _, _ = val.run(
                                        data=data_dict,
                                        task='test',
                                        batch_size=32,
                                        weights=os.path.join(configs.run_configs.ld_folder_name, 'seed_' + str(i), opt_val.weights),
                                        imgsz=opt_val.imgsz,
                                        iou_thres=0.65,
                                        single_cls=False,
                                        dataloader=test_loader,
                                        model=attempt_load(os.path.join(configs.run_configs.ld_folder_name, 'seed_' + str(i), opt_val.weights), opt_val.device).half(),
                                        save_dir=Path(os.path.join(opt_val.project, 'seed_' + str(i))),
                                        save_json=False,
                                        plots=True,
                                        callbacks=callbks,
                                        compute_loss=compute_loss,
                                        verbose=True,
                                        Round=round
                                        )

            round_results = pd.DataFrame([results])
            all_results = pd.concat([all_results, round_results])

            del(trainer)

        print(all_results)
        all_results.to_excel(cfg.run_configs.ld_folder_name + '/al_seed_yolov5_seed_' + str(i) + '.xlsx')


if __name__ == '__main__':
    def parse_opt_train(args, known=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=configs)
        parser.add_argument('--weights', type=str, default=args.detection.pretrained, help='initial weights path')
        parser.add_argument('--cfg', type=str, default=args.detection.yolocfg, help='model.yaml path')
        parser.add_argument('--data', type=str, default='./Models/detection/yolov5/data/focal.yaml', help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=args.detection.hyp,
                            help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=args.active_learning.max_epochs, help='total training epochs')
        parser.add_argument('--batch-size', type=int, default=16,
                            help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                            help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        parser.add_argument('--noplots', action='store_true', help='save no plot files')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram',
                            help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100,
                            help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                            help='Freeze layers: backbone=10, first3=0 1 2')
        parser.add_argument('--save-period', type=int, default=-1,
                            help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--seed', type=int, default=0, help='Global training seed')
        parser.add_argument('--local_rank', type=int, default=-1,
                            help='Automatic DDP Multi-GPU argument, do not modify')
        parser.add_argument('--save_dir', type=str, default=args.run_configs.ld_folder_name)

        # Logger arguments
        parser.add_argument('--entity', default=None, help='Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

        return parser.parse_known_args()[0] if known else parser.parse_args()

    def parse_val_opt(configs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=configs)
        parser.add_argument('--data', type=str, default=configs.detection.data, help='dataset.yaml path')
        parser.add_argument('--weights', nargs='+', type=str,
                            default=os.path.join('weights', 'best.pt'),
                            help='model path(s)')
        parser.add_argument('--batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
        parser.add_argument('--task', default='test', help='train, val, test, speed or study')
        parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--verbose', action='store_true', help='report mAP by class')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
        parser.add_argument('--project', default=configs.run_configs.ld_folder_name, help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.data = check_yaml(opt.data)  # check YAML
        opt.save_json |= opt.data.endswith('coco.yaml')
        opt.save_txt |= opt.save_hybrid
        print_args(vars(opt))
        return opt

    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='../../test.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.run_configs.gpu_id)
    opt_train = parse_opt_train(configs, True)
    opt_val = parse_val_opt(configs)
    main(configs, opt_train, opt_val)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
