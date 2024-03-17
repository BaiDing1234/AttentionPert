import re
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy

def extract_row(results_df, model, seed):
    model_dict = {
        'Ctrl': f'ctrl_mean_seed_{seed}_data_replogle_k562_essential',
        'CPA': f'CPA_seed_{seed}_data_replogle_k562_essential',
        'GEARS': f'GEARS_seed_{seed}_data_replogle_k562_essential',
        'Ours': f'Ours_seed_{seed}',
    }
    
    return results_df[results_df['model'] == model_dict[model]]

color_dict = {'Ctrl': 'royalblue', 
              'CPA': 'orange',
              'GEARS': 'red',
              'Ours': 'green'}

metrics_dict = {
        'unseen_single_frac_opposite_direction_top20_non_dropout': 'Percentage of top 20 DE genes\nwith opposite direction (Seen 0/1)',
        'unseen_single_frac_sigma_below_1_non_dropout': 'Percentage of top 20 DE genes\nbeyond one standard deviation (Seen 0/1)'}

def process_results(results_df):
    seeds = [1,2,3,4,5]
    new_dfs = {}
    new_df_default = {'model': ['Ctrl', 'CPA', 'GEARS', 'Ours']}

    #Generate seen0/1/2 and unseen_single mse percent for all splits:
    for seed in seeds:
        new_df = deepcopy(new_df_default)
        for m_k, m_v in metrics_dict.items():
            new_df[m_v] = ['' for _ in new_df_default['model']]
            ctrl_value = 0
            for i, model in enumerate(new_df_default['model']):
                row = extract_row(results_df, model, seed)
                value = float(row[m_k]) * 100
                value_std = float(row[f'{m_k}_std'])* 100
                if 'standard' in m_v:
                    value = 100 - value
                new_df[m_v][i] = '$' + f'{value:.1f}' + '_{\\text{ }\pm' + f'{value_std:.1f}' + '}$'
                if model == 'Ctrl':
                    ctrl_value = value
                    new_df[m_v][i] = '$' + f'{value:.1f}$'

        new_dfs[f'k562_split_{seed}_pod_psd'] = pd.DataFrame(new_df)


    for k, df in new_dfs.items():
        df.to_csv(f'csvs/{k}.csv')

csv_raw_file = 'csvs/k562_split_and_ctrl_manual.csv'
results_df = pd.read_csv(csv_raw_file)

process_results(results_df)


import matplotlib.pyplot as plt

# List of column names 'm' to plot
columns_to_plot = metrics_dict.values()  # Replace with your actual column names
columns_names = {k: ''.join(k.split('/')) for k in columns_to_plot}
#colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Colors for different models

# Function to parse the value and error bar
def parse_value_error(s):
    s = s.strip('$')
    s = s.rstrip('}')
    value = float(s.split('_')[0])
    if '\pm' in s:
        error = float(s.split('m')[1])
    else:
        error = 0
    return value, error

# Read and process data
data = {}
for i in range(1, 6):
    file_name = f'csvs/k562_split_{i}_pod_psd.csv'
    df = pd.read_csv(file_name)

    for column in columns_to_plot:
        if column not in data:
            data[column] = {'models': [], 'values': [], 'errors': []}

        for _, row in df.iterrows():
            value, error = parse_value_error(row[column])
            if value is not None:
                data[column]['models'].append(row['model'])
                data[column]['values'].append(value)
                data[column]['errors'].append(error)

# Plotting     
models = ["Ctrl", "CPA", "GEARS", "Ours"]
indices = np.arange(5)
fig, axs = plt.subplots(1, 2)
i = 0
j = 0
fig.set_size_inches(12, 14/3)
for column in columns_to_plot:
    ax = axs[j]
    max_ctrl = 0
    width = 1 / (len(models) + 1)
    for model_idx, model in enumerate(models):
        model_indices = [i for i, x in enumerate(data[column]['models']) if x == model]
        values = [data[column]['values'][i] for i in model_indices]
        errors = [data[column]['errors'][i] for i in model_indices]
        if model == 'Ctrl':
            max_ctrl = np.array(values).max()
        if i == 0 and j == 0:
            label = model
        else:
            label = None
        ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color= color_dict[model]) #, color=colors[model_idx % len(colors)])
    
    ax.set_xlabel('Split')
    ax.set_ylabel(f'{column}')
    #plt.title(f'Histogram of {column} by Model')
    ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
    ax.set_xticklabels(np.arange(1,6))
    if j == 1:
        ax.set_ylim(0, int(max_ctrl/10 + 2) * 10)
    if j == 1:
        i += 1
        j = 0
    else:
        j += 1
fig.legend(loc = 'upper right')
fig.tight_layout()
fig.savefig(f'fig/k562_pod_psd.pdf')
