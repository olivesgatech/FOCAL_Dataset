import os
import shutil
import pandas as pd

# This script was designed to be run on ws3 where the new focal data is located

def add_seq_images():
    df = pd.read_excel('./newseqs.xlsx', index_col=0, engine='openpyxl')

    root = '/data/ford/All_Time_Restricted_Sequences_Blurred'

    for _, seq in df.iterrows():
        if not os.path.exists(
                os.path.join(root, 'Images', 'deepen-' + seq['DatasetID'] + '-' + seq['DatasetName'], 'processed')):
            os.makedirs(
                os.path.join(root, 'Images', 'deepen-' + seq['DatasetID'] + '-' + seq['DatasetName'], 'processed'))


def move_new_seq_labels_to_correct_folder():
    root = '/data/ford/All_Time_Restricted_Sequences_Blurred'
    labels_root = '/data/ford/Final_Sequences_Time_Restricted_100_hours/Labels'
    all_new_labels = os.listdir(labels_root)
    all_new_labels.remove('tmp')
    all_new_labels.remove('zip')
    all_new_labels = [lbl.split('deepenLabels-')[1] for lbl in all_new_labels]

    df = pd.read_excel('./newseqs.xlsx', index_col=0, engine='openpyxl')

    for _, seq in df.iterrows():
        if seq['DatasetName'] in all_new_labels:
            shutil.move(os.path.join(labels_root, 'deepenLabels-' + seq['DatasetName'], seq['DatasetName'] + '.json'),
                        os.path.join(root, 'Labels',
                                     'Labels - ' + seq['DatasetID'] + ' - ' + seq['DatasetName'] + ' - default.json'))


def move_new_seq_images_to_correct_folder():
    old_image_root = '/data/ford/Final_Sequences_Time_Restricted_100_hours/Images/blurred'
    new_image_root = '/data/ford/All_Time_Restricted_Sequences_Blurred/Images'

    for root, subdirs, files in os.walk(old_image_root):
        for seq in subdirs:
            if seq.startswith('2021'):
                print('Yash')
                for subroot, subsubdirs, _ in os.walk(os.path.join(root, seq)):
                    for item in subsubdirs:  # 0000X or rosbag_analysis
                        if item.startswith('0'):
                            for frame in os.listdir(os.path.join(subroot, item, 'cam1')):
                                seqID = [s for s in os.listdir(new_image_root) if seq in s]
                                if len(seqID) == 1:
                                    # print(os.path.join(new_image_root, seqID[0], 'processed', frame))
                                    shutil.copy(os.path.join(subroot, item, 'cam1', frame),
                                                os.path.join(new_image_root, seqID[0], 'processed', frame))
                        break
                    break
            elif seq.startswith('deepen'):  # starts with deepen
                print('YEE')
                for subroot, subsubdirs, _ in os.walk(os.path.join(root, seq)):
                    for timeStamp in subsubdirs:
                        for frame in os.listdir(os.path.join(subroot, timeStamp, 'cam1')):
                            seqID = list(filter(seq.endswith, os.listdir(new_image_root)))
                            if len(seqID) == 1:
                                # print(os.path.join(new_image_root, seqID[0], 'processed', frame))
                                shutil.copy(os.path.join(subroot, timeStamp, 'cam1', frame),
                                            os.path.join(new_image_root, seqID[0], 'processed', frame))
                        break
                    break


def run_once_to_change_dir_tree():
    image_root = '/data/ford/All_Time_Restricted_Sequences_Blurred/Images'

    for seq in os.listdir(image_root):
        if seq.startswith('deepen-'):
            if not os.path.exists(os.path.join(image_root, seq, 'processed', 'images')):
                os.mkdir(os.path.join(image_root, seq, 'processed', 'images'))
                for frame in os.listdir(os.path.join(image_root, seq, 'processed')):
                    if frame.endswith('.jpg'):
                        shutil.move(os.path.join(image_root, seq, 'processed', frame),
                                    os.path.join(image_root, seq, 'processed', 'images', frame))


if __name__ == '__main__':
    # add_seq_images()
    # move_new_seq_labels_to_correct_folder
    # move_new_seq_images_to_correct_folder()
    run_once_to_change_dir_tree()
