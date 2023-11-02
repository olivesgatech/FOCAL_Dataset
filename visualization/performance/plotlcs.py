import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np


def get_acc_dataframe(target_path, nstart: int =128, nquery: int = 1024):
    """
    Returns aggregated dataframe with all runs stored with and additional column id
    Parameters:
        :param target_path: path to folder with target excel files
        :type target_path: str
    """

    # get all excel files
    path_list = glob.glob(target_path + '*.xlsx')
    total = pd.DataFrame([])

    # add excel files into complete dataframe
    for i in range(len(path_list)):
        df = pd.read_excel(path_list[i], index_col=0, engine='openpyxl')
        df['id'] = i
        rounds = df['id'].count()
        samples = [nstart + nquery * x for x in range(rounds)]
        df['Samples'] = np.array(samples)
        total = pd.concat([total, df])

    return total


def plot_lc(path_list, names, plotting_col='Test Acc', xlim=None, ylim=None, nstart: int = 128, nquery: int = 1024):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_acc_dataframe(target_path=path_list[i], nstart=nstart, nquery=nquery)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    total = total.rename_axis('Rounds')
    total.to_excel('blah2.xlsx')
    sns.lineplot(data=total, x='Samples', y=plotting_col, hue='Algorithm')
    plt.grid()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc="upper left", prop={'size': 10})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Sequences', fontsize=13)
    # plt.ylabel(plotting_col, fontsize=13)
    plt.ylabel(plotting_type, fontsize=13)
    plt.show()


if __name__ == "__main__":
    precursor = '~/PycharmProjects/Results/Ford-GATECH/frame/v6/'
    nstart = 128
    nquery = 3200
    plotting_type = 'mAP'
    # addons = ['al_entropy/', 'al_pcalentropy/', 'al_relsslentropy1x256/', 'al_relssentropy1x1024/']
    addons = [# 'al_entropy416_frame_pretrained/',
              'al_rand416_frame_pretrained/',
              # 'al_lconf416_sequence_pretrained/',
              # 'al_gauss416_sequence_pretrained/',
              # 'al_margin416_sequence_pretrained/'
              ]
    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [# 'Entropy Seq',
                 'Random Seq',
                 # 'LConf Seq ',
                 # 'GauSS Seq',
                 # 'Marg Seq'
                 ]
    plot_lc(path_list=paths, names=alg_names, plotting_col=plotting_type,
            # xlim=(128, 3000),
            # ylim=(0.08, 0.14),
            nstart=nstart,
            nquery=nquery)

