import argparse
import toml
import shutil
import os
import random
import numpy as np
import torch
from config import BaseConfig
from applications.detection.trainer import DetectionTrainer


def main(cfg: BaseConfig):
    trainer = DetectionTrainer(cfg)

    if cfg.run_configs.train:
        epochs = cfg.detection.epochs
        first = False
        for i in range(epochs):
            trainer.training(i, track_summaries=True)
            if cfg.run_configs.val:
                _ = trainer.validation(i)
            # if not first:
            #     first = True
            #     trainer.save_cache()

    if cfg.run_configs.test:
        _ = trainer.testing(0)

    # save learning dynamics
    # if not os.path.exists(cfg.run_configs.ld_folder_name):
    #     os.makedirs(cfg.run_configs.ld_folder_name)

    # if cfg.recognition.track_statistics:
    #     trainer.save_statistics(cfg.run_configs.ld_folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    # set random seeds deterministicly to 0
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    if configs.run_configs.train:
        shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.output_folder_name + '/parameters.toml')
        shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + '/parameters.toml')
