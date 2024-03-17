import scanpy as sc
from scipy import sparse
from scipy import stats
import numpy as np
import os
from collections import defaultdict
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from copy import deepcopy
import matplotlib.patches as mpatches


#set +o noclobber
#python NA_analysis.py > txt/NA_analysis_50.txt 2>&1

REPEAT = 5
ad = sc.read_h5ad("/home/ding.bai/pert_new/mywork/data/norman/perturb_processed.h5ad")

ctrl = ad[ad.obs.condition == 'ctrl'].X.toarray().mean(0) #N,
print("ctrl:", ctrl.shape)

DE_num = 200

conditions = ad.obs.condition.unique()
pert_genes = []
for condition in conditions:
    if condition == 'ctrl':
        continue
    cond1 = condition.split('+')[0]
    cond2 = condition.split('+')[1]
    if cond1 != 'ctrl': 
        if cond1 not in pert_genes:
            pert_genes.append(cond1)
    if cond2 != 'ctrl':
        if cond2 not in pert_genes:
            pert_genes.append(cond2)

num_genes = len(ad.var.gene_name)
if DE_num == 20:
    DE_dict_raw = ad.uns['top_non_dropout_de_20']
else:
    DE_dict_raw = ad.uns['rank_genes_groups_cov_all']
    for k, v in DE_dict_raw.items():
        DE_dict_raw[k] = v[:DE_num]
DE_dict = defaultdict(list)
symbol_to_idx = {symbol: i for i, symbol in enumerate(ad.var.gene_name.tolist())}
ens_to_idx = {ens: i for i, ens in enumerate(ad.var_names.tolist())}
for key, item in DE_dict_raw.items():
    symbols = key.split('_')[1].split('+')
    new_key = []
    for symbol in symbols:
        if symbol != 'ctrl':
            new_key.append(symbol_to_idx[symbol])
    new_key.sort()
    new_key = tuple(new_key)
    if new_key in DE_dict.keys():
        continue

    new_item = []
    for ens in item:
        new_item.append(ens_to_idx[ens])
    DE_dict[new_key] = np.array(new_item, dtype=int)
print("new DE dict: ", len(DE_dict.keys()), DE_dict[new_key])

   
pred_dir_dict = {"ours_NA": "preds/norman_beta_0.050_seed_1",
                 "ours_no_NA": "preds/AP_norman_beta_0.000_seed_1"} # beta = 0 means no NA at all.
methods = list(pred_dir_dict.keys())

method_pert_mean_dicts_dict = {}
perts_test = []
perts_0_1 = []
perts_0_2 = []
perts_1_2 = []
perts_2_2 = []
for method, pred_dir in pred_dir_dict.items():
    pert_mean_dicts = []

    for r in range(REPEAT):
        repeat_pred_dir = f"{pred_dir}_{r}"
        sp_files = os.listdir(repeat_pred_dir)
        pred_arrays = []
        for sp_file in sp_files:
            sparse_matrix = sparse.load_npz(f"{repeat_pred_dir}/{sp_file}")
            dense_matrix = sparse_matrix.toarray().reshape(-1, num_genes, 3) #B, N, 3
            pred_arrays.append(dense_matrix)
        pred_array = np.concatenate(pred_arrays, axis = 0) #All, N, 3
        print("pred_arrays.shape", pred_array.shape)
        del pred_arrays
        pert_rows = defaultdict(list)
        
        for i in range(pred_array.shape[0]):
            pert = tuple(np.where(pred_array[i, :, 0] > 0.5)[0])
            if len(pert) == 0:
                continue
            if len(pert) > 2:
                raise ValueError(f"len(pert) > 2! {pert}")
            pert_rows[pert].append(pred_array[i, :, 1:])
        del pred_array

        pert_mean_dict = {}
        for pert, rows in pert_rows.items():
            mean_row = np.mean(np.array(rows), axis=0) #pert_num_cells, N, 2 -> N, 2
            differ_row = np.zeros_like(mean_row)
            differ_row[:, 0] = mean_row[:, 1] - mean_row[:, 0] #y_true - y_pred
            differ_row[:, 1] = mean_row[:, 1] - ctrl  #y_true - ctrl
            pert_mean_dict[pert] = differ_row # N, 2 
            #print("pert:", pert, np.mean(differ_row[:, 0] ** 2), np.mean(differ_row[:, 1] ** 2))
            #print("pert:", pert, np.mean(differ_row[:, 0] ** 2)/np.mean(differ_row[:, 1] ** 2))
        pert_mean_dicts.append(pert_mean_dict)

        if method == "ours_NA" and r == 0:
            perts_test = list(pert_mean_dict.keys())
            unseen_genes = []
            for pert in perts_test:
                if len(pert) == 1:
                    perts_0_1.append(pert)
                    unseen_genes.append(pert[0])
            for pert in perts_test:
                if len(pert) == 2:
                    if len(np.intersect1d(pert, unseen_genes)) == 2:
                        perts_0_2.append(pert)
                    elif len(np.intersect1d(pert, unseen_genes)) == 1:
                        perts_1_2.append(pert)
                    elif len(np.intersect1d(pert, unseen_genes)) == 0:
                        perts_2_2.append(pert)
            print(f"Scenarios: 0/1, 0/2, 1/2, 2/2: {len(perts_0_1)}, {len(perts_0_2)}, {len(perts_1_2)}, {len(perts_2_2)}")
    method_pert_mean_dicts_dict[method] = pert_mean_dicts


scenarios = {"seen_0_1": perts_0_1, "seen_0_2": perts_0_2, "seen_1_2": perts_1_2, "seen_2_2": perts_2_2, "all2": perts_0_2 + perts_1_2 + perts_2_2}

def get_pert_vec_errors(pert, DE):
    pert_0 = ad.var.gene_name.tolist()[pert[0]]
    if len(pert) == 2:
        pert_1 = ad.var.gene_name.tolist()[pert[1]]
    else:
        pert_1 = "ctrl"
    pert_names = [f"{pert_0}+{pert_1}", f"{pert_1}+{pert_0}"]
    mat = []
    for pert_name in pert_names:
        mat.append(ad[ad.obs.condition == pert_name].X.toarray()[:, DE])
    mat = np.concatenate(mat, axis = 0)
    return mat.mean(0) - ctrl[DE] #, mat.std(0)

thres_list = [0.25, 0.5, 0.75]
csv_df = {'idx': [], "seen_0_2": [], "seen_1_2": [], "seen_2_2": []}
################
# create non-additive figures for seen_2_2. for only our method
################
for thres in thres_list:
    print(f"> NAG thres: {thres};")
    NAGs = []
    for name, scenario in scenarios.items():
        if name == "all2" or name == "seen_0_1":
            continue
        print(f">> Sceanario: {name};")
        loss_vec_dict = {'ours_no_NA': [[] for _ in range(REPEAT)],
                        'ours_NA': [[] for _ in range(REPEAT)]}
        for pert in scenario:
            DE = DE_dict[pert]
            genes = np.array([ad.var.gene_name.tolist()[i] for i in DE])
            pert_0 = ad.var.gene_name.tolist()[pert[0]]
            pert_1 = ad.var.gene_name.tolist()[pert[1]]
            pert_name = f"{pert_0}+{pert_1}"

            true_change_vec_multi= get_pert_vec_errors(pert, DE)
            true_change_vec_0= get_pert_vec_errors((pert[0], ), DE)
            true_change_vec_1= get_pert_vec_errors((pert[1], ), DE)
            add_change_vec = true_change_vec_0 + true_change_vec_1
            NAG = (np.abs((true_change_vec_multi - add_change_vec)/(true_change_vec_multi + 1e-9)) > thres)
            NAGs.append(np.sum(NAG).astype(np.float32))
            
            for r in range(REPEAT):
                for method in loss_vec_dict.keys():
                    ours_change_vec = method_pert_mean_dicts_dict[method][r][pert][DE, 1] - method_pert_mean_dicts_dict[method][r][pert][DE, 0]
                    ses = list((ours_change_vec[NAG] - true_change_vec_multi[NAG]) ** 2)
                    loss_vec_dict[method][r].extend(ses)
        for method in loss_vec_dict.keys():
            loss_vec_dict[method] = np.array(loss_vec_dict[method])
            print(f">>> Method: {method}")
            mse = loss_vec_dict[method].mean()
            std = loss_vec_dict[method].mean(1).std()
            print(f">>>> MSE: {mse:.4f}, std: {std:.4f}")
            if f'{method}_{thres}' not in csv_df['idx']:
                csv_df['idx'].append(f'{method}_{thres}')
            ele = '$' + f'{mse:.4f}' + '_{\\text{ }\pm' + f'{std:.4f}' + '}$'
            csv_df[name].append(ele)
    NAG_ratio = 100 * np.array(NAGs).mean()/DE_num
    print(f">> NAG_ratio: {NAG_ratio}")
csv_df = pd.DataFrame(csv_df)
csv_df.to_csv(f'csvs/NAG_results_{DE_num}.csv')
            
