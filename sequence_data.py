import numpy as np
from PIL import Image
import torch
from torch.utils import data

from sequence_data_interface import seqDataset_interface


class seqDataset(data.Dataset):
    def __init__(self, split, transform=None): 
        """
        :param split: 'train' | 'test'.
        """
        self.split = split
        self.transform = transform
        imdb = seqDataset_interface()
        data_seq = imdb.generate_data_sequence(self.split)

        self.data = self.get_data(data_seq) 
    
    def __getitem__(self, index):
        bbox = self.data['bbox'][index]
        image = Image.open(self.data['image'][index]).convert('RGB')
        im_crop = image.crop(tuple(bbox))
        if self.transform is not None:
            im_crop =self.transform(im_crop)
        label = self.data['label'][index] 
        oid = self.data['id'][index] 
        ret = {'image':im_crop, 'label':label, 'oid':oid, 'sid':index}
                
        return ret

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
            if k=='bbox': 
                tracks = np.asarray(tracks)
                tracks[:,2] = tracks[:,0] + tracks[:,2]
                tracks[:,3] = tracks[:,1] + tracks[:,3]
            ret[k] = tracks
        
        return ret    
    
    
if __name__ == '__main__':
    dataset = seqDataset('train')
