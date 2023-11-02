import os
import seaborn as sns
import pandas as pd
import glob
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

COST_EVENT_PATH = 'Scene_Labels_Updated_New3_TimeLabels_Added.xlsx'
QUERY_EVENT_PATH = 'FocalBoxCountEstimatesCost.xlsx'


def get_cost_dataframe(target_path, nstart: int =128, nquery: int = 1024, query_type: str = 'Sequences',
                       num_interpolations: int = 5):
    """
    Returns aggregated dataframe with all runs stored with and additional column id
    Parameters:
        :param target_path: path to folder with target excel files
        :type target_path: str
    """

    # get all excel files
    path_list = glob.glob(target_path + 'round*')
    print(target_path)
    # total = pd.DataFrame(columns=['ID', 'Round', 'Samples', 'Hours'])
    total_dict = defaultdict(list)
    prev = defaultdict(lambda: defaultdict(int))
    seen = set()
    event_list = np.load(target_path + 'event_list.npy')
    count = 0
    # with open(COST_EVENT_PATH, 'rb') as f:
    #     cost_dict = pickle.load(f)
    cost_event_df = pd.read_excel(COST_EVENT_PATH, engine='openpyxl')
    missed_paths = []
    for i in range(len(path_list)):
        seed_paths = glob.glob(path_list[i] + '/seed*')

        for j in range(len(seed_paths)):
            npy_path = seed_paths[j] + '/new_events.npy'
            events = np.load(npy_path)
            print(len(events))
            prevk = events
            events = np.char.rpartition(events, '-')[:, -1].tolist()
            costs = 0
            for e in events:
                row = cost_event_df.loc[cost_event_df['DatasetID'] == e]
                try:
                    if query_type == 'Sequences':
                        costs += row['Days'].values[0]*24 + row['Hours'].values[0]
                    elif query_type == 'Frames':
                        effective_labeled_frames = row['Frames'].values[0]//num_interpolations
                        costs += (row['Days'].values[0]*24 + row['Hours'].values[0])/effective_labeled_frames
                        k = 0
                    else:
                        raise Exception('Query type not implemented yet!')
                except:
                    missed_paths.append(e)

            samples = nstart + i*nquery
            total_dict['ID'].append(j)
            total_dict['Rounds'].append(i)
            total_dict['Samples'].append(samples)
            if j not in prev.keys() or i-1 not in prev[j].keys():
                prev[j][i-1] = 0
            cur_cost = prev[j][i-1] + costs
            total_dict['Hours'].append(cur_cost)
            prev[j][i] = cur_cost

    total = pd.DataFrame(data=total_dict)
    print(f'Missed Events:\n{set(missed_paths)}')

    return total

def plot_lc(path_list, names, plotting_col='Test Acc', xlim=None, ylim=None, nstart: int = 128, nquery: int = 1024,
            query_type: str = 'Sequences', num_interpolations: int = 5, seeds=None):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    total = total.rename_axis('Rounds')
    sns.lineplot(data=total, x='Samples', y=plotting_col, hue='Algorithm', errorbar=('se', 0.4), legend=False)
    sns.lineplot(data=total, x='Samples', y='Theoretical Min Cost', linestyle='-.', legend='full')
    sns.lineplot(data=total, x='Samples', y='Theoretical Max Cost', linestyle='-.', legend='full')

    plt.grid()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.ylim(0, 800)
    plt.legend(loc="best", prop={'size': 10})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(query_type, fontsize=13)
    plt.ylabel(plotting_type, fontsize=13)
    labels = alg_names
    labels.append('Lower Bound Cost')
    labels.append('Upper Bound Cost')
    plt.legend(loc='best', labels=labels)
    plt.show()


def get_cost(target_path, query_type, seeds):
    cost_event_df = pd.read_excel(COST_EVENT_PATH, engine='openpyxl')
    train_df = cost_event_df.loc[cost_event_df['Split'] == 'train']
    missed_paths = []
    total_dict = defaultdict(list)
    theory = pd.DataFrame([])
    theoreticals = pd.DataFrame([])
    path_list = glob.glob(os.path.join(target_path, 'seed*'))  # seed_0, seed_1, seed_2, ...
    path_list.sort()
    for i in range(0):
        costs = 0
        theory_min_cost = train_df.head(12)
        theory_min_cost = list(theory_min_cost['Train Theory Min Cost'])
        theory_min_cost.insert(0, 0.0)
        theory_max_cost = train_df.tail(12)
        theory_max_cost = list(theory_max_cost['Train Theory Max Cost'])
        theory_max_cost.reverse()
        theory_max_cost.insert(0, 0.0)
        theory['Theoretical Min Cost'] = theory_min_cost
        theory['Theoretical Max Cost'] = theory_max_cost
        total_dict['ID'].append('initial')
        total_dict['Seed'].append(i)
        total_dict['Round'].append(0)
        total_dict['Samples'].append(0)
        total_dict['Cost (Hours)'].append(0.0)
        for j in range(12):  # only using 12 rounds
            seed_paths = glob.glob(os.path.join(path_list[i], 'round' + str(j)))  # round0, round1, round2, ....
            npy_path = os.path.join(seed_paths[0], 'seed' + str(i), 'new_events.npy')
            events = np.load(npy_path)
            events = np.char.rpartition(events, '-')[:, -1].tolist()
            for e in events:
                if e.endswith('_'):
                    e = e[:-1]
                row = cost_event_df.loc[cost_event_df['DatasetID'] == e]
                try:
                    if query_type == 'Sequences':
                        costs += row['Full Hours'].values[0]
                    else:
                        raise Exception('Query type not implemented yet!')
                except:
                    missed_paths.append(e)

            samples = nstart + j*nquery
            total_dict['ID'].append(events[0])
            total_dict['Seed'].append(i)
            total_dict['Round'].append(j)
            total_dict['Samples'].append(samples)
            total_dict['Cost (Hours)'].append(costs)  # cumulative cost

        theoreticals = pd.concat([theoreticals, theory], ignore_index=True)
    total = pd.DataFrame(data=total_dict)
    total['Theoretical Min Cost'] = theoreticals['Theoretical Min Cost']
    total['Theoretical Max Cost'] = theoreticals['Theoretical Max Cost']
    return total


def get_map(target_path, seeds):
    total_dict = defaultdict(list)
    path_list = glob.glob(os.path.join(target_path, 'seed*'))
    for i in range(seeds):
        total_dict['Seed'].append(i)
        total_dict['Round'].append(0)
        total_dict['Map'].append(0.0)
        for j in range(12):  # only using 12 rounds
            map_df = pd.read_csv(os.path.join(path_list[i], 'test_results_round_' + str(j) + '.txt'),
                                 header=None)
            #  columns = total_images, total_objects_in_class, precision, recall, mAP_0.5, mAP_0.5:0.95
            total_dict['Seed'].append(i)
            total_dict['Round'].append(j)
            total_dict['Map'].append(map_df.iloc[0, 5])
    total = pd.DataFrame(data=total_dict)
    return total


def plot_cost_vs_map(path_list, names):
    if len(path_list) != len(names):
        raise Exception("Each element in the path list must have a corresponding name")
    total = pd.DataFrame([])
    for i in range(len(path_list)):
        cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        cost_df['Algorithm'] = names[i]
        map_df = get_map(target_path=path_list[i], seeds=seeds)
        cost_df['mAP'] = map_df['Map']
        total = pd.concat([total, cost_df], axis=0)

    palette_tab10 = sns.color_palette("tab10", 10)
    palette = [palette_tab10[0], palette_tab10[1], palette_tab10[2], palette_tab10[3], palette_tab10[4]]
    sns.lineplot(data=total, x='Cost (Hours)', y='mAP', hue='Algorithm', errorbar=('se', 0.5), palette=palette)
    plt.axvline(x=485, color=palette_tab10[3], ls='--')
    plt.axvline(x=521, color=palette_tab10[2], ls='--')
    plt.axvline(x=574, color=palette_tab10[4], ls='--')
    plt.axvline(x=583, color=palette_tab10[0], ls='--')
    plt.axvline(x=734, color=palette_tab10[1], ls='--')
    plt.xlim(0.0, 750)
    plt.ylim(0.0, 0.7)
    plt.title('Conformal Sampling')
    plt.legend(loc='center')
    plt.grid()
    plt.show()


def plot_cost_vs_map_ryan(path_list, names):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")
    total = pd.DataFrame([])
    total_map = pd.DataFrame([])
    for i in range(len(path_list)):
        cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        cost_df['Algorithm'] = names[i]
        map_df = get_map(target_path=path_list[i], seeds=seeds)
        cost_df['mAP'] = map_df['Map']
        total = pd.concat([total, cost_df], axis=0)
        total = pd.concat([total, cost_df], axis=0)
        total_map = pd.concat([total_map, map_df], axis=0)

        cost_df['Inv. Cost'] = 1 / cost_df['Cost (Hours)']
        cost_df['Mean Inv. Cost'] = cost_df.groupby(['Samples', 'Algorithm'])['Inv. Cost'].transform('mean')
        cost_df['Mean mAP'] = cost_df.groupby(['Samples', 'Algorithm'])['mAP'].transform('mean')
    total.reset_index(drop=True, inplace=True)
    total_map.to_excel('blah2.xlsx')
    total['Inv. Cost'] = 1/total['Cost (Hours)']
    total['Cost (Hours)'] = total.groupby(['Samples', 'Algorithm'])['Cost (Hours)'].transform('mean')
    palette_tab10 = sns.color_palette("tab10", 10)
    palette = [palette_tab10[5], palette_tab10[6], palette_tab10[7], palette_tab10[8], palette_tab10[9], 'blue']
    sns.lineplot(data=total, x='Cost (Hours)', y='mAP', hue='Algorithm', palette=palette)
    palette_tab10 = sns.color_palette("tab10", 10)
    plt.axvline(x=598, color=palette_tab10[7], ls='--')
    plt.axvline(x=644, color=palette_tab10[6], ls='--')
    plt.axvline(x=660, color=palette_tab10[8], ls='--')
    plt.axvline(x=666, color=palette_tab10[5], ls='--')
    plt.axvline(x=565, color=palette_tab10[9], ls='--')  # coreset
    plt.axvline(x=685, color='blue', ls='--')  # badge
    plt.xlim(0.0, 750)
    plt.ylim(0.0, 0.7)
    plt.title('Inferential Sampling')
    plt.legend(loc='center')
    plt.grid()
    plt.show()


def plot_map_vs_query(path_list, queries, names):
    query_df = pd.read_excel(QUERY_EVENT_PATH, engine='openpyxl')
    query_scores = []
    total = pd.DataFrame([])
    for i in range(len(path_list)):
        cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        cost_df['Algorithm'] = names[i]
        map_df = get_map(target_path=path_list[i], seeds=seeds)
        cost_df['Map'] = map_df['Map']
        for seq in cost_df['ID']:
            query = query_df["SequenceID"].str.endswith(seq)
            tmp = query_df[query]
            if queries == 'Motion':
                query_scores.append(int(tmp[queries].values))
            elif queries == 'BoxCountEstimate':
                query_scores.append(int(tmp[queries].values))
        cost_df[queries] = query_scores
        total = pd.concat([total, cost_df], axis=0)
    # total['ThereticalLeastFrame'] =
    # sns.lineplot(x=total[queries], y=total['Map'])
    sns.regplot(x=total[queries], y=total['Map'], data=total, ci=None, line_kws={'color': 'red'})
    plt.xlabel(queries)
    plt.ylabel('mAP')
    plt.show()


def plot_cost_vs_query(path_list, queries, names):
    query_df = pd.read_excel(QUERY_EVENT_PATH, engine='openpyxl')
    query_scores = []
    noncumulative_cost = []
    total = pd.DataFrame([])
    for i in range(len(path_list)):
        cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        cost_df['Algorithm'] = names[i]
        map_df = get_map(target_path=path_list[i], seeds=seeds)
        cost_df['Map'] = map_df['Map']
        for seq in cost_df['ID']:
            query = query_df["SequenceID"].str.endswith(seq)
            tmp = query_df[query]
            if queries == 'Motion':
                query_scores.append(int(tmp[queries].values))
            elif queries == 'BoxCountEstimate':
                query_scores.append(int(tmp[queries].values))
            noncumulative_cost.append(float(tmp['cost'].values))
        cost_df[queries] = query_scores
        cost_df['NonCumulativeCost'] = noncumulative_cost
        total = pd.concat([total, cost_df], axis=0)
    # sns.scatterplot(x=total[queries], y=total['NonCumulativeCost'])
    sns.regplot(x=total[queries], y=total['NonCumulativeCost'], data=total, ci=None, line_kws={'color': 'red'})
    plt.xlabel(queries)
    plt.ylabel('Cost')
    plt.show()


def cost_vs_map_budgets(path_list, names, budget, budget_values):
    from scipy import integrate

    aucs = []
    for b_val in budget_values:
        for i in range(len(path_list)):
            total = pd.DataFrame([])
            cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
            cost_df['Algorithm'] = names[i]
            map_df = get_map(target_path=path_list[i], seeds=seeds)
            cost_df['mAP'] = map_df['Map']
            total = pd.concat([total, cost_df], axis=0)

            total = total.loc[total[budget] <= b_val]
            rand_seeds = total['Seed'].unique()
            auc = 0
            for seed in rand_seeds:
                tmp_total = total.loc[total['Seed'] == seed]
                auc += integrate.trapz(x=tmp_total['Cost (Hours)'], y=tmp_total['mAP'])
                aucs.append([names[i], auc/len(rand_seeds), b_val, seed])
            # aucs.append([names[i], auc/len(rand_seeds), b_val, seed])
            # a = 1
    df = pd.DataFrame(aucs, columns=['Algorithm', 'AUC', budget, 'seed'])
    palette_tab10 = sns.color_palette("tab10", 10)
    # palette = [palette_tab10[0], palette_tab10[1], palette_tab10[2], palette_tab10[3], palette_tab10[4]]
    palette = [palette_tab10[5], palette_tab10[6], palette_tab10[7], palette_tab10[8], palette_tab10[9], 'blue']
    sns.lineplot(data=df, x=budget, y='AUC', hue='Algorithm', palette=palette, errorbar=('se', 0.30))
    if budget == 'mAP':
        plt.title('Performance-Focused Budget')
        plt.xlim(0.4, 0.6)
        plt.ylabel('PAR')
    else:
        plt.title('Cost-Focused Budget')
        plt.xlim(0, 200)  # 200
        plt.ylabel('CAR')
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylim(0.0, 200)  # 200
    plt.show()

def compute_slope(path_list, names, bin_count):
    from scipy.stats import linregress
    import random
    import decimal
    total = pd.DataFrame([])
    for i in range(len(path_list)):
        cost_df = get_cost(target_path=path_list[i], query_type=query_type, seeds=seeds)
        cost_df['Algorithm'] = names[i]
        map_df = get_map(target_path=path_list[i], seeds=seeds)
        cost_df['mAP'] = map_df['Map']
        total = pd.concat([total, cost_df], axis=0)

    slopes = []
    for alg in names:
        sub_total = total.loc[total['Algorithm'] == alg]
        sub_total['bin_cost'] = pd.qcut(sub_total['Cost (Hours)'], q=bin_count, labels=range(bin_count))
        for bin in sub_total['bin_cost'].unique():
            bucket = sub_total.loc[sub_total['bin_cost'] == bin]
            if bucket['Cost (Hours)'].mean() == bucket['Cost (Hours)'].iloc[0]:
                collect = []
                for cost in bucket['Cost (Hours)']:
                    const = float(random.randrange(1, 100))/1e2
                    cost += const
                    collect.append(cost)
                bucket['Cost (Hours)'] = collect

            slope, _, _, _, _ = linregress(x=bucket['Cost (Hours)'], y=bucket['mAP'])
            slopes.append([alg, bin, slope])
    df = pd.DataFrame(slopes, columns=['Algorithm', 'bin', 'slope'])
    print('mean slope is:')
    print(df.groupby(['Algorithm']).mean())
    print('max slope is:')
    print(df.groupby(['Algorithm']).max())


if __name__ == "__main__":
    precursor = '../../results/sequence/yolov5n/'
    nstart = 2
    nquery = 1
    seeds = 2
    plotting_type = 'Cost (Hours)'
    query_type = 'Sequences'
    interpolation_rate = 5

    addons = [
              # 'al_leastframe_seq_pretrained/',
              # 'al_mostframe_seq_pretrained/',
              # 'al_minmotion_seq_pretrained/',
              # 'al_minmaxmotion_seq_pretrained/',
              # 'al_minboxes_seq_pretrained/',
            'al_random_seq_pretrained/',
            'al_false_seq_pretrained_mean_combine/',
            'al_entropy_seq_pretrained/',
            'al_gauss_seq_pretrained',
            'al_coreset_seq_pretrained_mean_combine',
            'al_badge_seq_pretrained_mean_combine'
              ]

    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [
                 # 'Least Frame',
                 # 'Most Frame',
                 # 'Min Motion',
                 # 'Min Max Motion',
                 # 'Min Boxes',
                'Random',
                'FALSE',
                'Entropy',
                'Gauss',
                'Coreset',
                'BADGE'
                 ]

    # plot_lc(path_list=paths, names=alg_names, plotting_col=plotting_type,
    #         nstart=nstart, nquery=nquery, query_type=query_type,
    #         num_interpolations=interpolation_rate, seeds=seeds)

    # plot_cost_vs_map(path_list=paths, names=alg_names)
    plot_cost_vs_map_ryan(path_list=paths, names=alg_names)
    # plot_map_vs_query(path_list=paths, names=alg_names, queries='BoxCountEstimate')
    # plot_cost_vs_query(path_list=paths, queries='BoxCountEstimate', names=alg_names)
    # get_cost_dataframe(target_path=paths, nstart=nstart, nquery=nquery, query_type=query_type,num_interpolations=interpolation_rate)
    # cost_vs_map_budgets(path_list=paths, names=alg_names, budget='mAP', budget_values=np.arange(0.4, 0.65, 0.1))
    # cost_vs_map_budgets(path_list=paths, names=alg_names, budget='Cost (Hours)', budget_values=np.arange(0.0, 250.0, 100.0))
    # compute_slope(path_list=paths, names=alg_names, bin_count=3)

