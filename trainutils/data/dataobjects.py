from torch.utils.data import DataLoader
from config import BaseConfig


class DatasetStructure:
    def __init__(self, cfg: BaseConfig):
        # different sets
        self.train_set = None
        self.test_set = None
        self.val_set = None

        # set statistics
        self.train_len = None
        self.test_len = None
        self.num_classes = None
        self.img_size = None
        self.in_channels = None
        self.is_configured = False

        self.cfg = cfg


class LoaderObject:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader,
                 data_configs: DatasetStructure):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.data_config = data_configs

    def save_cache(self):
        if not self.data_config.is_configured:
            raise Exception('Dataset not configured yet!')
        self.data_config.train_set.save_cache()
        self.data_config.val_set.save_cache()
        self.data_config.test_set.save_cache()

        self.train_loader = DataLoader(self.data_config.train_set,
                                       batch_size=self.data_config.cfg.detection.batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(self.data_config.val_set,
                                     batch_size=self.data_config.cfg.detection.batch_size,
                                     shuffle=False)
        self.test_loader = DataLoader(self.data_config.test_set,
                                      batch_size=self.data_config.cfg.detection.batch_size,
                                      shuffle=False)
