import numpy as np
from PIL import Image
import torch
from torch.utils import data

from data.loader.seqdatainterface import SeqDatasetInterface
from config import BaseConfig

LABELS = {
    'Pedestrian': 0,
    'Pedestrian_With_Object': 0,
    'Wheeled_Pedestrian': 0,
    'Bicycle': 1,
    'Motorcycle': 2,
    'Car': 3,
    'Truck': 4,
    'Trailer': 4,
    'Bus': 4,
    'Semi_truck': 4,
    'Stroller': 5,
    'Trolley': 5,
    'Cart': 5,
    'Emergency_Vehicle': 6,
    'Construction_Vehicle': 6,
    'Towed_Object': 6,
    'Vehicle_Towing': 6,
    'Animal': 7,
    'Trash Cans': 8,
    'Traffic_cones': 8,
}

import time
class SeqDataset(data.Dataset):
    def __init__(self, split, cfg: BaseConfig, transform=None):
        """
        :param split: 'train' | 'test'.
        """
        self._cfg = cfg
        self.split = split
        self.transform = transform
        imdb = SeqDatasetInterface(cfg)
        data_seq = imdb.generate_data_sequence(self.split)

        self.data = self.get_data(data_seq)
        self.num_classes = len(LABELS.keys())

        self.img_loading_cum = 0.0
        self.label_cum = 0.0
        self.label_map_cum = 0.0
        self.count = 0

    def __getitem__(self, index):
        bbox = self.data['bbox'][index]
        image = Image.open(self.data['image'][index]).convert('RGB')
        start = time.time()
        im_crop = image.crop(tuple(bbox))
        end = time.time()

        if self.transform is not None:
            im_crop =self.transform(im_crop)
        self.img_loading_cum += (end - start)

        start = time.time()
        label = LABELS[self.data['label'][index]]
        end = time.time()
        self.label_map_cum += (end - start)

        self.count += 1
        if self.count % 2 == 0:
            self.count = 0
            frac1 = self.img_loading_cum/2.0
            frac2 = self.label_map_cum / 2.0
            self.img_loading_cum = 0.0
            self.label_map_cum = 0.0
            #print('IMG PROC TIME: %f' % frac1)
            #print('LABEL MAP TIME: %f' % frac2)
        oid = self.data['id'][index] 
        # ret = {'image': im_crop, 'label': label, 'oid': oid, 'sid': index}
                
        # return ret
        return im_crop, label, oid, index

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])    

    def get_data(self, data):
        """
        :param data: The raw data
        :return: A dictionary containing training and testing data
        """
        ret = {}
        for k in data.keys(): 
            tracks = []
            for track in data[k]:
                tracks.extend(track)
            if k == 'bbox':
                tracks = np.asarray(tracks)
                tracks[:, 2] = tracks[:, 0] + tracks[:, 2]
                tracks[:, 3] = tracks[:, 1] + tracks[:, 3]
            ret[k] = tracks
        
        return ret    
    
    
if __name__ == '__main__':
    dataset = SeqDataset('train')
