"""
Interface for the sequence dataset 
"""
import sys
sys.path.append('../../../Ford-GATECH/')
import pickle
import json
import numpy as np
from os.path import join, abspath, exists
from os import listdir, makedirs
import glob
from tqdm import tqdm
from operator import itemgetter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from config import BaseConfig
import toml, os
# set of mobile objects to use. To be changed
MOBILE_AGENTS = {'Pedestrian_With_Object',
 'Pedestrian',
 'Car',
 'Bicycle',
 'Truck',
 'Bus',
 'Semi_truck',
 'Motorcycle'
                 }

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


    # 'Animal': 8,
    # 'Trash Cans': 8,
    # 'Traffic_cones': 8,

# TODO: ALWAYS UPDATE THIS WHEN YOU CHANGE THE DATASET CLASSES
NUM_CLASSES = 4

class SeqDatasetInterface:
    def __init__(self, cfg: BaseConfig, regen_pkl=False):
        # paths 
        self.data_path = cfg.data.data_loc if cfg.data.data_loc else self._get_default_path()
        self._seq_path = join(self.data_path, 'Images/blurred/')
        self._label_path = join(self.data_path, 'Labels/')
        self._data_split_ids_path = join(self.data_path, 'split_ids')
        ## the tokens could be used as ids of the sequences
        self._token_to_label_file = {}
        self._token_to_seq_folder = {}
        self._date_time_to_token = {}
#         self._token_to_date_time = {}
        ## initiate tokens to match label files and the sequence folders
        self._init_tokens()  # Note: only call it once
        self.savetoken_label = self._token_to_label_file
        self.savetoken_seq = self._token_to_seq_folder
        self.savetoken_time = self._date_time_to_token
        self.video_ids = self._get_video_ids('all')
        self._regen_pkl = regen_pkl
        self.corrupted = {}
        self._cfg = cfg
        
    def _init_tokens(self):
        list_label_files = glob.glob(join(self._label_path, '*.json')) # all ~180 label files
        list_seq_folders = glob.glob(join(self._seq_path, 'deepen*'))
        for label_file in list_label_files:
            item = label_file.split('-')
            _token, _date_time = item[1].strip(), '-'.join(item[2:-1]).strip()
            assert _token not in self._token_to_label_file, f'duplicate token {_token} in label files!'
            self._token_to_label_file[_token] = label_file
            assert _date_time not in self._date_time_to_token, f'duplicate date&time {_date_time} in label files!'
            self._date_time_to_token[_date_time] = _token
#             self._token_to_date_time[_token]=_date_time
        for seq_folder in list_seq_folders:
            _token = seq_folder.split('/')[-1].split('-')[-1]
            assert _token not in self._token_to_seq_folder, f'duplicate token {_token} in sequence folders!'
            self._token_to_seq_folder[_token] = seq_folder
        #TODO: check the sequences that do not have matched label files
        for _token in self._token_to_seq_folder.keys():    
            if _token not in self._token_to_label_file: 
                print(f'token {_token}, sequence "{self._token_to_seq_folder[_token]}" does not have label file')    
#                 print(f'token {_token}, sequence "{self._token_to_date_time[_token]}" does not have label file')    
                
    # ====================== paths ====================== 
    @property
    def cache_path(self):
        """
        Generate a path to save cache files
        :return: Cache file folder path
        """
        cache_path = os.path.expanduser(join('~/', self._cfg.run_configs.cache_dir))
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path    
    
    def _get_default_path(self):
        """
        Return the default data root path where files are expected to be placed.
        :return: the default path to the dataset folder
        """
        return f'/data/ford/active_learning/'
    
    # ====================== statistics ======================
    def _get_video_ids(self, image_set):
        """
        Returns a list of all video ids after selection
        :return: The list of video ids (date-time)
        """
        vid_ids = [] 
        vid_id_file = join(self._data_split_ids_path, f'{image_set}.txt')
        with open(vid_id_file, 'rt') as fid:
            vid_ids.extend([x.strip() for x in fid.readlines()])

        missing = []
        for vid in vid_ids:
            if vid not in self._date_time_to_token.keys():
                missing.append(vid)
                assert vid in self._date_time_to_token, f'{vid} does not have label file'
        # print(sorted(self._date_time_to_token.keys()))
        print(len(self._date_time_to_token.keys()))
        print('Missing IDs:')
        print(sorted(missing))
        print(len(missing))
        return vid_ids

    def _get_frame_count(self, vid):
        """
        Returns the total number of frames of a sequence
        :return: int value of frame count
        """
        list_img_files = glob.glob(join(self._token_to_seq_folder[self._date_time_to_token[vid]], 'processed/images/*.jpg'))
        list_img_files.sort()
        return len(list_img_files)
    
    def _get_frame_resolution(self, vid):
        path_img_file = join(self._token_to_seq_folder[self._date_time_to_token[vid]], 'processed/images/00000.pcd.jpg')
        return  plt.imread(path_img_file).shape[:2]

    # ====================== database ======================    
    def generate_database(self, filename):
        """
        Generate a database by integrating all annotations
        Dictionary structure:
        'vid_id'(str): {
            'num_frames': int
            'width': int
            'height': int
            'Pedestrian'(str): {
                'ped_id'(str): {
                    'frames': list(int)
                    'bbox': list([x1, y1, x2, y2])
                    'attributes'(str): {
            'Car'(str): {
                'vehicle_id' (str): {
                    'frames': list(int)
                    'bbox': list([x1, y1, x2, y2])
                    'attributes'(str): {
            ...
        :return: A prediction database dictionary
        """
        print('---------------------------------------------------------')
        print("Generating database") 
        cache_file = join(self.cache_path, filename)
        if exists(cache_file) and not self._regen_pkl:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('sequence database loaded from {}'.format(cache_file))
            return database
        
        database = {}
        for vid in self.video_ids:
            print('Getting annotations for %s' % vid)
            try: 
                database[vid] = self._get_all_annotations_obj(vid)
            
                database[vid]['num_frames'] = self._get_frame_count(vid)  
                database[vid]['height'], database[vid]['width'] = self._get_frame_resolution(vid)
            except: 
                print(f'can not get the images or labels for sequnce "{vid}"!')
                continue
        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database
    
    def generate_data_sequence(self, image_set, filename='tmp_database.pkl'):
        """
        :param image_set: 'train' | 'test'.
        :return: 
        """
        print('---------------------------------------------------------')
        print(f"Generating {image_set} sequence data")
        annotations = self.generate_database(filename)
#         sequence = self._get_trajectories(image_set, annot_database, **params)
        num_objs = 0
        image_seq, box_seq = [], []
        lbls_seq, ids_seq = [], []
          
        video_ids = self._get_video_ids(image_set) 
        
        for vid in video_ids:
            if 'num_frames' not in annotations[vid]: continue
            for cls, obj_annots in annotations[vid].items():
                if cls in ['num_frames', 'height', 'width']: continue
                # iterate through each object in this class
                for oid in obj_annots.keys(): 
                    num_objs += 1
                    frame_ids = obj_annots[oid]['frames']
                    images = [join(self._token_to_seq_folder[self._date_time_to_token[vid]], f'processed/images/{fid:05d}.pcd.jpg') for fid in obj_annots[oid]['frames']]
                    boxes = obj_annots[oid]['bbox']
                    ids = [f'{vid}/{fid}/{oid}' for fid in frame_ids]
                    image_seq.append(images)
                    box_seq.append(boxes)
                    lbls_seq.append([cls] * len(boxes))
                    ids_seq.append(ids)

        print('Split: %s' % image_set)
        print('Number of objects: %d ' % num_objs)

        return {'image': image_seq,
                'label': lbls_seq,
                'bbox': box_seq,
                'id': ids_seq
               }        
        return sequence    
    
    # detection generator 
    def get_data_detection(self, image_set, method, filename, **params):
        """
        Generates data for detection algorithms
        :param image_set: Split set name
        :param method: Detection algorithm: centernet, yolo
        :param file_path: Where to save the script file
        :return: Object samples
        """
        file_path = abspath(join(self.cache_path, 'detection/'))
        if not exists(file_path):
            makedirs(file_path)
        annotations = self.generate_database(filename)
        video_ids = self._get_video_ids(image_set)
        obj_samples = {}
        total_sample_count = 0
        for vid in video_ids:
            try:
                num_frames = annotations[vid]['num_frames']
            except:
                continue
            for i in range(num_frames): #TODO: sanity check on potential missing frames
                obj_samples[join(self._token_to_seq_folder[self._date_time_to_token[vid]],
                                   f'processed/images/{i:05d}.pcd.jpg')] = []
            for cls, obj_annots in annotations[vid].items():
                if cls in ['num_frames', 'height', 'width']: continue
                if cls not in LABELS.keys():
                    # print(f'{cls} not in labels!')
                    continue
                for oid in obj_annots.keys():
                    imgs = [join(self._token_to_seq_folder[self._date_time_to_token[vid]],
                                   f'processed/images/{fid:05d}.pcd.jpg') for fid in obj_annots[oid]['frames']]
                    boxes = obj_annots[oid]['bbox']
                    for i, b in enumerate(boxes):
                        if imgs[i] not in obj_samples:
                            continue
                        obj_samples[imgs[i]].append({'box': b, 'label': LABELS[cls], 'id': f'{vid}/{obj_annots[oid]["frames"][i]}/{oid}'})
                        total_sample_count += 1
        print('Number of samples %d ' % total_sample_count)
        if method == 'yolo':
            return self._generate_csv_yolo(image_set, file_path, obj_samples)
        # elif method == 'centernet':
        #     return self._generate_csv_centernet(image_set, file_path, obj_samples)
        else:
            raise NotImplementedError

    # ====================== alutils ======================
    def _get_by_frame(self, fn):
        def _gbf(json_row):
            return int(json_row['file_id'].replace('.pcd','')) == fn
        return _gbf
    
    def _get_by_obj(self, obj):
        def _gbo(json_row):
            return int(json_row['label_id']==obj)
        return _gbo
    
    def _get_by_mobile_agent(self): ## pre-selecting mobile agents
        def _gbma(json_row):
            return int(json_row['label_category_id'] in MOBILE_AGENTS)
        return _gbma
    
    def _box_only(self, json_row):
        return json_row['label_type'] == "box"
    
    def _threed_box_only(self, json_row):
        return json_row['label_type'] == "3d_bbox"  ## key needs to be changed
    
    def visualize_frame(self, vid, fid, label_2d_bbox):
        path_to_img = join(self._token_to_seq_folder[self._date_time_to_token[vid]], f'processed/images/{fid:05d}.pcd.jpg')
        img = plt.imread(path_to_img)
        fig, ax = plt.subplots(1,1,figsize=(30,20))
        ax.imshow(img)
        for box in label_2d_bbox:
            u,v,w,h = box
            ax.add_patch(mpatches.Rectangle((u,v),w,h,linewidth=4,edgecolor='blue',facecolor='blue', alpha=0.2))
        return ax
    
    # ====================== annotations ======================    
    def _get_annotations(self, _labels):
        """get 2d detection annotations"""
        _labels_2d = list(filter(self._box_only, _labels))  # 2d boxes
        _labels_2d = sorted(_labels_2d, key=lambda k: int(k['file_id'].replace('.pcd',''))) # ensure frame ascending order
        
        annotations = {}
        annotations['frames'] = list(map(lambda k: int(k['file_id'].replace('.pcd','')), _labels_2d))
        annotations['bbox']= list(map(itemgetter('box'), _labels_2d)) # Note: list([x1, y1, w, h])
        annotations['obj_ids'] = list(map(itemgetter('label_id'), _labels_2d))  
        #TODO: append 'attributes' to the annotations
#         annotations['attributes']: []   # list(int) 
        return annotations

    def _get_all_annotations_obj(self, vid):
        """collect object-wise labels"""
        path_to_file = self._token_to_label_file[self._date_time_to_token[vid]]  ## vid-->token-->label file 
        assert exists(path_to_file), f'{path_to_file} not exist!'
        with open(path_to_file, 'r') as f:
            labels=(json.load(f))['labels']
        database = {}
#         labels = list(filter(self._get_by_mobile_agent(), labels))
        list_objs = (np.unique(list(map(itemgetter('label_id'), labels)))).tolist()
        for obj in list_objs:
            _labels_curr_obj = list(filter(self._get_by_obj(str(obj)), labels))   ## extracts single object information   
            annotations = self._get_annotations(_labels_curr_obj)  ##   
            obj_category = obj.split(':')[0]
            if obj_category not in database:
                database[obj_category] = {}
            database[obj_category][str(obj)] = annotations     
        return database

    def _get_all_annotations_frame(self, vid):
        """collect frame-wise labels (bboxes, object ids, image?)"""
        path_to_file = self._token_to_label_file[self._date_time_to_token[vid]]  ## vid-->token-->label file 
        assert exists(path_to_file), f'{path_to_file} not exist!'
        with open(path_to_file, 'r') as f:
            labels=(json.load(f))['labels']
        database = []
        num_frames = self._get_frame_count(vid)  
        ## Note: iterating each frame is slow
        for fid in tqdm(range(num_frames)):
            _labels_curr_frame = list(filter(self._get_by_frame(fid), labels))  
            annotations=self._get_annotations(_labels_curr_frame) 
            # visualize single frame
#             ax = self.visualize_frame(vid, fid, annotations['bbox'])
            database.append(annotations)  ## index of the database list is the frame index         
        return database
    
    def _generate_csv_yolo(self, image_set, file_path, obj_samples):
        """
        CSV data generation for YOLO
        :param image_set: Data split
        :param file_path: Path to save the data
        :param obj_samples: Dictionary of all samples
        """
        data_save_path = join(file_path, 'yolo_' + image_set + '.txt')
        cor_save_path = join(file_path, 'yolo_corrupted_' + image_set + '.txt')
        corrupted = {}
        # remove corrupted files
        print('Removing corrupted files')
        count = 0
        if exists(cor_save_path):
            os.remove(cor_save_path)
        for img, samples in sorted(obj_samples.items()):
            count += 1
            try:
                tmp = Image.open(img)
            except:
                #f.write('%s ' % (img))
                #for s in samples:
                #    box = s['box']
                #    f.write('%.0f,%.0f,%.0f,%.0f,%.0f ' % (box[0], box[1], box[2], box[3], s['label']))
                #f.write('\n')
                corrupted[img] = True

        print('Writing to txt')
        uncorrupted = 0
        if not exists(data_save_path):
            with open(data_save_path, "wt") as f:
                for img, samples in sorted(obj_samples.items()):
                    if not samples:
                        continue
                    if img not in corrupted.keys():
                        uncorrupted += 1
                        written = False
                        # print(img.split('/')[5])
                        for s in samples:
                            box = s['box']
                            if box[2] > 50 and box[3] > 75:
                                if not written:
                                    written = True
                                    f.write('%s ' % (img))
                                f.write('%.0f,%.0f,%.0f,%.0f,%.0f ' % (box[0], box[1], box[2], box[3], s['label']))
                            else:
                                l = s['label']
                                # print(f'Skipping label of h {box[2]} and w {box[3]} and class {l}')
                        if written:
                            f.write('\n')

                print('Data generated for YOLO')
        self.corrupted[image_set] = list(corrupted.keys())
        print('#Corrupted in ' + image_set)
        print(len(list(corrupted.keys())))
        print('#Uncorrupted in ' + image_set)
        print(uncorrupted)
        print('Total:')
        print(count)
        return data_save_path


if __name__ == '__main__':
    import argparse
    import toml
    import os
    from data import get_yolo_dataset
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='../../example_config.toml')
    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)

    data_interface = SeqDatasetInterface(cfg=configs)
    with open('tokens_time_mapping.pkl', 'wb') as f:
        pickle.dump(data_interface.savetoken_time, f)
    with open('tokens_label_mapping.pkl', 'wb') as f:
        pickle.dump(data_interface.savetoken_label, f)
    with open('tokens_seq_mapping.pkl', 'wb') as f:
        pickle.dump(data_interface.savetoken_seq, f)
    database = data_interface.generate_database('tmp_database.pkl')
    splits = ['train', 'val', 'test']
    for split in splits:
        data_interface.get_data_detection(image_set=split, method='yolo', filename='tmp_database.pkl')

    # loaders = get_yolo_dataset(configs, debug=True)
    # tbar = tqdm(loaders.val_loader)
    # for i, sample in enumerate(tbar):
    #     x = 0

    # corrupted = loaders.data_config.val_set.corrupted
    # print(corrupted)
    #print('Corrupted FIlES:')
    #for split in ['val']:
    #    print(split)
    #    print(data_interface.corrupted[split])


