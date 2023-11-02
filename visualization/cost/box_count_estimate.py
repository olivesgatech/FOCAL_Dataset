from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow
import pandas as pd
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from visualization.spatiotemporal import plot_statistics


def generate_flow_maps():
    # Specify the path to model config and checkpoint file
    config_file = 'utils/pwcnet_ft_4x1_300k_kitti_320x896.py'
    checkpoint_file = 'utils/pwcnet_ft_4x1_300k_kitti_320x896.pth'

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')

    # test image pair, and save the results
    image_root = '/data/ford/All_Time_Restricted_Sequences_Blurred/Images'

    box_count_estimates = []
    all_sequences = []
    for root, subdirs, files in os.walk(image_root):
        for seq in subdirs:
            print(seq)
            frames = sorted(os.listdir(os.path.join(root, seq, 'processed', 'images')))
            first_frame_accessed = False
            prev_frame = None
            save_dir = os.path.join(root, seq, 'processed', 'images', 'flow_maps')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            box_count_estimate = 0
            for frame in frames:
                if frame.endswith('.jpg'):
                    if first_frame_accessed:
                        result = inference_model(model, os.path.join(root, seq, 'processed', 'images', prev_frame),
                                                 os.path.join(root, seq, 'processed', 'images', frame))
                        flow_map = visualize_flow(result, save_file=os.path.join(save_dir, prev_frame[:-4] + '-' +
                                                                                 frame[:-4] + '_flow_map.png'))
                        img = cv2.imread(os.path.join(save_dir, prev_frame[:-4] + '-' +
                                                      frame[:-4] + '_flow_map.png'), cv2.IMREAD_GRAYSCALE)
                        blurred = cv2.GaussianBlur(img, (7, 7), 0)
                        threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                        n, _ = cv2.connectedComponents(threshold.astype('uint8'), connectivity=8)
                        box_count_estimate += (n - 1)
                    else:
                        first_frame_accessed = True
                    prev_frame = frame
            all_sequences.append(seq)
            box_count_estimates.append(box_count_estimate)

    df = pd.DataFrame([], columns=['SequenceID', 'BoxCountEstimate'])
    df['SequenceID'] = all_sequences
    df['BoxCountEstimate'] = box_count_estimates
    df.to_excel("./FocalBoxCountEstimates.xlsx")


def complute_box_and_motion_estimates():
    image_root = '/data/ford/All_Time_Restricted_Sequences_Blurred/Images'

    box_count_estimates = []
    all_sequences = []
    all_motion = []
    count = 0
    for root, subdirs, files in os.walk(image_root):
        for seq in subdirs:
            if seq.startswith('deepen'):
                print(seq + ' count: ' + str(count))
                path = os.path.join(root, seq, 'processed', 'images', 'flow_maps')
                box_count_estimate = 0
                motion = 0
                flow_maps = sorted(os.listdir(path))
                for flow_map in flow_maps:
                    try:
                        img = cv2.imread(os.path.join(path, flow_map), cv2.IMREAD_GRAYSCALE)
                        blurred = cv2.GaussianBlur(img, (7, 7), 0)
                        threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                        n, _ = cv2.connectedComponents(threshold.astype('uint8'), connectivity=8)
                        box_count_estimate += (n - 1)  # excluded background from the estimate
                        motion += sum(sum(img))
                    except:
                        print('Exception handled: ' + os.path.join(path, flow_map))
                        box_count_estimate += 0
                        motion += 0
                all_sequences.append(seq)
                box_count_estimates.append(box_count_estimate/len(flow_maps))  # average box count estimate
                all_motion.append(motion)
                count += 1

    df = pd.DataFrame([], columns=['SequenceID', 'AvgBoxCountEstimate', 'Motion'])
    df['SequenceID'] = all_sequences
    df['AvgBoxCountEstimate'] = box_count_estimates
    df['Motion'] = all_motion
    df.to_excel("./FocalBoxCountMotionEstimates.xlsx")


def visualize_estimates():
    df_si_ti_time = pd.read_excel('Scene_Labels_Updated_New3_TimeLabels_Added.xlsx')
    df_estimates = pd.read_excel('FocalBoxCountMotionEstimates.xlsx')
    df_siti = plot_statistics.plot_hours_versus()

    all_stats = pd.DataFrame([], columns=['SequenceID', 'Motion', 'AvgBoxCountEstimate', 'avg_si', 'cost'])
    costs = []
    frames = []
    avg_si = []
    avg_ti = []
    density = []
    boxes = []

    for seq in df_estimates['SequenceID']:
        costs.append(float(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Full Hours'].values))
        frames.append(float(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Frames'].values))
        density.append(float(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Label Density'].values))
        boxes.append(int(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Boxes'].values))
        avg_si.append(float(df_siti.loc[df_siti['SequenceID'] == seq, 'avg_si'].values))
        avg_ti.append(float(df_siti.loc[df_siti['SequenceID'] == seq, 'avg_ti'].values))


    all_stats['SequenceID'] = df_estimates['SequenceID']
    all_stats['Motion'] = df_estimates['Motion']
    all_stats['Frames'] = frames
    all_stats['ObjectDensity'] = density
    all_stats['AvgBoxCountEstimate'] = df_estimates['AvgBoxCountEstimate']
    all_stats['Boxes'] = boxes
    all_stats['BoxCountEstimate'] = df_estimates['BoxCountEstimate']
    all_stats['cost'] = costs
    all_stats['avg_si'] = avg_si
    all_stats['avg_ti'] = avg_ti
    all_stats['box_est/si'] = all_stats['AvgBoxCountEstimate']/all_stats['avg_si']
    all_stats['si/box_est'] = all_stats['avg_si']/all_stats['AvgBoxCountEstimate']
    all_stats['motion/si'] = all_stats['Motion']/all_stats['avg_si']
    all_stats = all_stats.sort_values('cost')
    all_stats.to_excel('FocalBoxCountEstimatesCost.xlsx')


    # sns.pairplot(all_stats, kind="reg")

    # all_stats = all_stats.loc[all_stats['Frames'] > 300]
    # plt.ticklabel_format(axis='y', style='sci')
    # all_stats = all_stats.loc[all_stats['Boxes'] < 100000]
    sns.scatterplot(y=all_stats['Boxes'], x=all_stats['cost'])
    g = sns.regplot(y=all_stats['Boxes'], x=all_stats['cost'], data=all_stats, ci=50, line_kws={'color': 'red'}, scatter=False)

    slope, intercept, r, p, sterr = scipy.stats.linregress(x=g.get_lines()[0].get_xdata(),
                                                           y=g.get_lines()[0].get_ydata())

    print(slope)
    g.axes.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    plt.title('Number of Boxes and Cost', fontsize=13)
    plt.ylabel('Number of Boxes', fontsize=13)
    plt.xlabel('Cost', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid()
    plt.show()



# complute_box_and_motion_estimates()
visualize_estimates()