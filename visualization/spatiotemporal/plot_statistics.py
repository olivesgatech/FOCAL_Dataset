from typing import Generator, Optional
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from skimage import feature
import seaborn as sns
from scipy import ndimage
import numpy as np
import pandas as pd
import json
import os


def plot_spatia_temporal(statistic):
    root = '/data/ford/All_Time_Restricted_Sequences_Blurred/Images'  # '../../data/FOCAL/blurred'
    sequences = os.listdir(os.path.join(root))

    all_si_statistic = []
    all_ti_statistic = []

    for seq in sequences:
        if not seq.startswith('.') and not seq.endswith('.sh'):
            jsonFile = os.path.join(root, seq, 'processed', 'images', seq + '.json')
            if os.path.getsize(jsonFile) > 0:
                f = open(jsonFile)
                siti = json.load(f)
                all_si_statistic.append(siti[statistic + '_si'])
                all_ti_statistic.append(siti[statistic + '_ti'])
                print((siti[statistic + '_si'], siti[statistic + '_ti'], seq))
                f.close()

    plt.scatter(all_si_statistic, all_ti_statistic)
    plt.xlabel('Spatial Information (SI)')
    plt.ylabel('Temporal Information (TI)')
    plt.title(statistic.upper() + ' TI vs ' + statistic.upper() + ' SI for FOCAL Dataset')
    plt.grid()
    plt.savefig(statistic + '.png')


def plot_hours_versus():
    df_si_ti_time = pd.read_excel('Scene_Labels_Updated_New3_TimeLabels_Added.xlsx')
    root = '/media/yash-yee/Storage/All_Time_Restricted_Sequences_Blurred/Images'  # '../../data/FOCAL/blurred'
    sequences = os.listdir(os.path.join(root))

    stats = ['avg_si', 'avg_ti', 'si', 'ti', 'std_si', 'std_ti']

    all_avgsi_statistic = []
    all_avgti_statistic = []
    all_stdsi_statistic = []
    all_stdti_statistic = []
    all_ti_statistic = []
    all_si_statistic = []
    cost = []
    all_seqs = []
    all_seq_boxes = []
    scene_ID = []

    for seq in sequences:
        if not seq.startswith('.') and not seq.endswith('.sh'):
            jsonFile = os.path.join(root, seq, 'processed', 'images', seq + '.json')
            f = open(jsonFile)
            siti = json.load(f)
            all_avgsi_statistic.append(siti[stats[0]])
            all_avgti_statistic.append(siti[stats[1]])
            all_si_statistic.append(siti[stats[2]])
            all_ti_statistic.append(siti[stats[3]])
            all_stdsi_statistic.append(siti[stats[4]])
            all_stdti_statistic.append(siti[stats[5]])
            scene_ID.append(int(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Scene ID'].values))
            cost.append(float(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Full Hours'].values))
            all_seqs.append(seq)
            all_seq_boxes.append(float(df_si_ti_time.loc[df_si_ti_time['Sequence Name'] == seq, 'Boxes'].values))
            f.close()

    all_stats = pd.DataFrame([], columns=['SequenceID', 'SceneID', 'avg_si', 'avg_ti', 'std_si', 'std_ti', 'cost'])
    all_stats['SequenceID'] = all_seqs
    all_stats['SceneID'] = scene_ID
    all_stats['avg_si'] = all_avgsi_statistic
    all_stats['avg_ti'] = all_avgti_statistic
    # all_stats['ti'] = all_ti_statistic
    # all_stats['si'] = all_si_statistic
    all_stats['cost'] = cost
    all_stats['std_si'] = all_stdsi_statistic
    all_stats['std_ti'] = all_stdti_statistic
    # all_stats['boxes'] = all_seq_boxes
    all_stats = all_stats.sort_values('cost')

    return all_stats
    # all_stats['boxes/avg_si'] = all_stats['boxes']/all_stats['avg_si']

    # plt.figure(1)
    # plt.title('Comparison of Cost vs Avg SI')
    # sns.set_theme()
    # sns.set(font_scale=1.2)
    # sns.regplot(x=all_stats['avg_si'], y=all_stats['cost'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(x=all_stats['avg_si'], y=all_stats['cost'])
    # plt.xlabel('Avg SI')
    # plt.ylabel('Cost')
    #
    # plt.figure(2)
    # plt.title('Comparison of Cost vs Avg TI')
    # sns.set_theme()
    # sns.set(font_scale=1.2)
    # sns.regplot(x=all_stats['avg_ti'], y=all_stats['cost'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(x=all_stats['avg_ti'], y=all_stats['cost'])
    # plt.xlabel('Avg TI')
    # plt.ylabel('Cost')
    #
    # plt.figure(10)
    # plt.title('Comparison of Cost vs Avg TI/SI')
    # sns.set_theme()
    # sns.set(font_scale=1.2)
    # sns.regplot(x=all_stats['avg_ti']/all_stats['avg_si'], y=all_stats['cost'], data=all_stats, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(x=all_stats['avg_ti']/all_stats['avg_si'], y=all_stats['cost'])
    # plt.xlabel('Avg TI/SI')
    # plt.ylabel('Cost')
    # #
    # plt.figure(4)
    # plt.title('Comparison of Cost vs abs(Avg SI - TI)')
    # sns.set_theme()
    # sns.set(font_scale=1.2)
    # sns.regplot(x=abs(all_stats['avg_si'] - all_stats['avg_ti']), y=all_stats['cost'], data=all_stats, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(x=abs(all_stats['avg_si'] - all_stats['avg_ti']), y=all_stats['cost'])
    # plt.xlabel('Avg SI/TI')
    # plt.ylabel('Cost')

    # plt.figure(9)
    # plt.title('TI/SI Distributions')
    # sns.set_style('darkgrid')
    # cheapest_seq = all_stats['7']
    # for ti, si, cost in zip(all_stats['ti'], all_stats['si'], all_stats['cost']):
    #     if cost < 42:
    #         sns.distplot(np.divide(ti, si), label=cost, color='red')
    #     elif cost > 84:
    #         sns.distplot(np.divide(ti, si), label=cost, color='blue')
    #     else:
    #         sns.distplot(np.divide(ti, si), label=cost, color='green')
    # plt.figure(1)
    # plt.title('Comparison of Cost vs Number of Objects/Avg SI')
    # # sns.set_theme()
    # # sns.set(font_scale=1.2)
    # sns.regplot(y=all_stats['boxes']/all_stats['avg_si'], x=all_stats['cost'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'}, lowess=True)
    # sns.scatterplot(y=all_stats['boxes']/all_stats['avg_si'], x=all_stats['cost'])
    # plt.ylabel('Number of Objects/Avg SI')
    # plt.xlabel('Cost')
    #
    # sns.pairplot(all_stats[['cost', 'avg_si', 'avg_ti', 'boxes', 'boxes/avg_si']], kind="reg")

    # plt.figure(2)
    # plt.title('Comparison of Cost vs Number of Objects/Avg TI')
    # sns.regplot(y=all_stats['boxes']/all_stats['avg_ti'], x=all_stats['cost'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(y=all_stats['boxes']/all_stats['avg_ti'], x=all_stats['cost'])
    # plt.ylabel('Number of Objects/Avg TI')
    # plt.xlabel('Cost')
    #
    # plt.figure(5)
    # plt.title('Comparison of Objects and Cost')
    # sns.regplot(y=all_stats['boxes'], x=all_stats['cost'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(y=all_stats['boxes'], x=all_stats['cost'])
    # plt.ylabel('Number of Objects')
    # plt.xlabel('Cost')
    # #
    # plt.figure(6)
    # plt.title('Comparison of SI and Number of Objects')
    # sns.regplot(y=all_stats['boxes'], x=all_stats['avg_si'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(y=all_stats['boxes'], x=all_stats['avg_si'])
    # plt.ylabel('Number of Objects')
    # plt.xlabel('Avg SI')
    #
    # plt.figure(7)
    # plt.title('Comparison of TI and Number of Objects')
    # sns.regplot(y=all_stats['boxes'], x=all_stats['avg_ti'], data=all_stats, ci=None, scatter_kws={'color': 'black'},
    #             line_kws={'color': 'red'})
    # sns.scatterplot(y=all_stats['boxes'], x=all_stats['avg_ti'])
    # plt.ylabel('Number of Objects')
    # plt.xlabel('Avg TI')

    # plt.figure(7)
    # plt.title('SI Distributions')
    # sns.set_style('darkgrid')
    # for si, cost in zip(all_stats['si'], all_stats['cost']):
    #     if cost < 42:
    #        sns.distplot(si, label=cost, color='red')
    #     elif cost > 84:
    #        sns.distplot(si, label=cost, color='blue')
    #     else:
    #        sns.distplot(si, label=cost, color='green')

    # plt.figure(8)
    # plt.title('TI/SI Distributions')
    # sns.set_style('darkgrid')
    # for ti, si, cost in zip(all_stats['ti'], all_stats['si'], all_stats['cost']):
    #     if cost < 42:
    #         sns.distplot(np.divide(ti, si), label=cost, color='red')
    #     elif cost > 84:
    #         sns.distplot(np.divide(ti, si), label=cost, color='blue')
    #     else:
    #         sns.distplot(np.divide(ti, si), label=cost, color='green')
    # plt.show()


def predict_cost():
    all_data = plot_hours_versus()
    exclude = [1, 2, 5, 6, 7, 19, 26, 27, 28, 29, 30, 32, 33, 36, 37, 38, 39, 40]  # scene ID for test and val set
    train = [36, 37, 38, 39, 40]  # scene ID for val set being used as train set here
    df_train = all_data[all_data['SceneID'].isin(train)]
    x_train = df_train.iloc[:, 2:6]
    y_train = df_train['cost']

    df_test = all_data[~all_data['SceneID'].isin(exclude)]
    x_test = df_test.iloc[:, 2:6]
    y_test = df_test['cost']

    clf = ensemble.GradientBoostingRegressor(n_estimators=2000, max_depth=2, min_samples_split=3,
                                             learning_rate=0.0001, loss='huber', min_samples_leaf=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(r2_score(y_test, y_pred))


# plot_spatia_temporal('max')
# plot_hours_versus()
# predict_cost()