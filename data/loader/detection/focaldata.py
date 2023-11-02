import os
import shutil
import random

# Four classes total
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


def convert_focal_to_yolov5(file):
    f = open(os.path.join('focaldata_timerestricted', file), 'r')
    Split = file.split('_')[3]
    if not os.path.exists(os.path.join('../../../Models/detection/yolov5/data/focal_timerestricted/images', Split[:-4])):
        os.mkdir(os.path.join('../../../Models/detection/yolov5/data/focal_timerestricted/images', Split[:-4]))
    if not os.path.exists(os.path.join('../../../Models/detection/yolov5/data/focal_timerestricted/labels', Split[:-4])):
        os.mkdir(os.path.join('../../../Models/detection/yolov5/data/focal_timerestricted/labels', Split[:-4]))

    # if not os.path.exists(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', file[:-4])):
    #     os.mkdir(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', file[:-4]))
    # if not os.path.exists(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/labels', file[:-4])):
    #     os.mkdir(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/labels', file[:-4]))

    for line in f.readlines():
        data = line.split(' ')
        num_objects = len(data) - 1
        seq = data[0].split('/')[5]
        frame = data[0].split('/')[8]
        # im_path = os.path.join('/media/yash-yee/Storage/FOCAL/blurred', data[0].split('blurred/')[1])
        im_path = os.path.join('/data/ford/All_Time_Restricted_Sequences_Blurred', 'Images', data[0].split('Images/')[1])
        # im_save = os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', file[:-4],
        #                        seq + '_' + frame)
        im_save = os.path.join('/home/yash-yee/projects/Ford-GATECH/Models/detection/yolov5/data/focal_timerestricted/images', Split[:-4],
                               seq + '_' + frame)
        if not os.path.isfile(im_save):
            shutil.copy(im_path, im_save)
        # im_save = os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/labels', file[:-4],
        #                        seq + '_' + frame)
        im_save = os.path.join('/home/yash-yee/projects/Ford-GATECH/Models/detection/yolov5/data/focal_timerestricted/labels', Split[:-4],
                               seq + '_' + frame)

        if not os.path.isfile(im_save):
            file_object = open(os.path.join(im_save[:-3] + 'txt'), 'w')

            for i in range(1, num_objects):
                # obj_lbl = data[i].split(',')[5].split('/')[2].split(':')[0]
                lbl = int(data[i].split(',')[4])  # LABELS[obj_lbl]
                x = int(data[i].split(',')[0])  # / FRAME_WIDTH
                y = int(data[i].split(',')[1])  # / FRAME_HEIGHT
                width = int(data[i].split(',')[2])  # / FRAME_WIDTH
                height = int(data[i].split(',')[3])  # / FRAME_HEIGHT

                # Convert to yolo format for ultralytics yolov5 code
                im_h, im_w = 2048, 2448
                x_center = (x + width / 2) / im_w
                y_center = (y + height / 2) / im_h
                norm_width = width / im_w
                norm_height = height / im_h

                file_object.write('\n'.join([str(lbl) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' +
                                             str(norm_width) + ' ' +
                                             str(norm_height) + '\n']))
            file_object.close()


# Do not run this function!
def create_val_split():
    if not os.path.exists(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', 'val')):
        os.mkdir(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', 'val'))
    if not os.path.exists(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/labels', 'val')):
        os.mkdir(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/labels', 'val'))

    current_root = '/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/'
    train_list = os.listdir(os.path.join('/media/yash-yee/Storage/Yolo/yolov5/data/focal_ultralytics/images', 'train'))
    count = len(train_list)
    val_idxs = random.sample(range(0, count), int(0.1 * count))
    for idx in val_idxs:
        shutil.move(os.path.join(current_root, 'images', 'train', train_list[idx]),
                    os.path.join(current_root, 'images', 'val', train_list[idx]))
        shutil.move(os.path.join(current_root, 'labels', 'train', train_list[idx][:-3] + 'txt'),
                    os.path.join(current_root, 'labels',  'val', train_list[idx][:-3] + 'txt'))



convert_focal_to_yolov5('yolo_focal_timerestricted_val.txt')
convert_focal_to_yolov5('yolo_focal_timerestricted_test.txt')
convert_focal_to_yolov5('yolo_focal_timerestricted_train.txt')


