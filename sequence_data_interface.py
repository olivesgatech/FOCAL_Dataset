"""
Interface for the sequence dataset 
"""
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

# set of mobile objects to use. To be changed
MOBILE_AGENTS = {'Pedestrian_With_Object',
 'Pedestrian',
 'Car',
 'Bicycle',
 'Truck',
 'Bus',
 'Semi_truck',
 'Motorcycle',
                }

class seqDataset_interface():
    def __init__(self, data_path='', regen_pkl=False):
        # paths 
        self.data_path = data_path if data_path else self._get_default_path()    
        self._seq_path = join(self.data_path, 'tmp/')
        self._label_path = join(self.data_path, 'FordLabels/')
        self._data_split_ids_path = join(self.data_path, 'split_ids')
        ## the tokens could be used as ids of the sequences
        self._token_to_label_file = {}
        self._token_to_seq_folder = {}
        self._date_time_to_token = {}
#         self._token_to_date_time = {}
        ## initiate tokens to match label files and the sequence folders
        self._init_tokens()  # Note: only call it once
        self.video_ids = self._get_video_ids('all')
        self._regen_pkl = regen_pkl
        
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
        cache_path = abspath(join(self.data_path, 'data_cache'))
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
        for vid in vid_ids:
            assert vid in self._date_time_to_token, f'{vid} does not have label file'
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

    # ====================== utils ======================    
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
    
    
    
    
    
if __name__ == '__main__':
    data_interface = seqDataset_interface()
    data_seq = data_interface.generate_data_sequence('train', filename='tmp_database.pkl')

    