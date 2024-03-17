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

REPEAT = 5


ad = sc.read_h5ad("../data/norman/perturb_processed.h5ad")
with open("../data/gene2go_all.pkl", "rb") as f:
    g2g = pickle.load(f)
with open("../data/essential_all_data_pert_genes.pkl", "rb") as f:
    essential_pert = pickle.load(f)
g2g = {i: g2g[i] for i in essential_pert if i in g2g}

ctrl = ad[ad.obs.condition == 'ctrl'].X.toarray().mean(0) #N,
print("ctrl:", ctrl.shape)

essential_pert = list(np.unique(list(g2g.keys())))
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
for pert_gene in pert_genes:
    if pert_gene not in essential_pert:
        essential_pert.append(pert_gene)
essential_pert = np.array(essential_pert)
print("essential_pert len: ", len(essential_pert))
print("essential_pert[-10:]: ", essential_pert[:10], essential_pert[-10:])

num_genes = len(ad.var.gene_name)

DE_dict_raw = ad.uns['top_non_dropout_de_20']
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

if not os.path.exists("csvs/norman_go_20.csv"):
    go_graph = pd.read_csv("../data/norman/go.csv")
    gene_degrees = np.arange(num_genes)
    new_graph = pd.DataFrame({"source": [], "target": [], "importance": []})
    for symbol, idx in tqdm(symbol_to_idx.items()):
        symbol_graph = go_graph[go_graph["source"] == symbol]
        weights = symbol_graph["importance"]
        weights = np.array(weights)
        if len(weights) <= 20:
            min_weight = 0
        else:
            weights.sort()
            min_weight = weights[-20]
        new_graph = pd.concat([new_graph, symbol_graph[symbol_graph["importance"] >= min_weight]])
    new_graph.to_csv("csvs/norman_go_20.csv")

go_graph = pd.read_csv("csvs/norman_go_20.csv")
gene_degrees = np.arange(num_genes)
for symbol, idx in tqdm(symbol_to_idx.items()):
    weights = go_graph[go_graph["source"] == symbol]["importance"]
    if len(weights) > 0:
        gene_degrees[idx] = np.array(weights).sum()
    else:
        gene_degrees[idx] = 0
print("go_graph max, min degree: ", gene_degrees.max(), gene_degrees.min()) 

pert_graph_dict = {}
for pert in DE_dict.keys():
    pert_nodes = []
    for pi in pert:
        pi_symbol = ad.var.gene_name.tolist()[pi]
        for target in go_graph[go_graph["source"] == pi_symbol]["target"].tolist():
            pert_nodes.append(symbol_to_idx[target])
        for source in go_graph[go_graph["target"] == pi_symbol]["source"].tolist():
            pert_nodes.append(symbol_to_idx[source])
    pert_nodes = np.unique(pert_nodes).astype(int)
    pert_graph_dict[pert] = pert_nodes

   
pred_dir_dict = {"ours": "preds/norman_beta_0.050_seed_1",
                 "gears": "preds/GEARS_seed_1_data_norman"}
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
            if method == "ours":
                pert = tuple(np.where(pred_array[i, :, 0] > 0.5)[0])
                if len(pert) == 0:
                    continue
                if len(pert) > 2:
                    raise ValueError(f"len(pert) > 2! {pert}")
                pert_rows[pert].append(pred_array[i, :, 1:])
            elif method == "gears":
                pert_old = tuple(pred_array[i, :2, 0])
                if len(pert_old) == 0:
                    continue
                if len(pert_old) > 2:
                    raise ValueError(f"len(pert) > 2! {pert_old}")
                pert_new = []
                for pi in pert_old:
                    if int(pi) >= 1:
                        gi = essential_pert[int(pi) - 1]
                        pert_new.append(symbol_to_idx[gi])
                pert_new.sort()
                pert_rows[tuple(pert_new)].append(pred_array[i, :, 1:])
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

        if method == "gears" and r == 0:
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
import seaborn as sns





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


################
# create non-additive figures for seen_2_2. for only our method, for only "FOXA1+HOXB9"
################

the_one_pert = "FOXA1+HOXB9"
the_gene_idxs = np.array([0,1,3,4,6,8,11,15,16])

for name, scenario in scenarios.items():
    if name == "all2" or name == "seen_0_1":
        continue
    print(f"Sceanario: {name};")
    if not os.path.exists(f'non_add_figs/{name}'):
        os.mkdir(f'non_add_figs/{name}')
    for pert in scenario:
        DE = DE_dict[pert]
        genes = np.array([ad.var.gene_name.tolist()[i] for i in DE])
        pert_0 = ad.var.gene_name.tolist()[pert[0]]
        pert_1 = ad.var.gene_name.tolist()[pert[1]]
        pert_name = f"{pert_0}+{pert_1}"
        if pert_name != the_one_pert:
            continue
        
        #true_change_vec_multi = method_pert_mean_dicts_dict["ours"][0][pert][DE, 1]
        #true_change_vec_0 = method_pert_mean_dicts_dict["ours"][0][(pert[0],)][DE, 1]
        #true_change_vec_1 = method_pert_mean_dicts_dict["ours"][0][(pert[1],)][DE, 1]
        #true_change_vec_multi, true_change_errors_multi = get_pert_vec_errors(pert, DE)
        #true_change_vec_0, true_change_errors_0 = get_pert_vec_errors((pert[0], ), DE)
        #true_change_vec_1, true_change_errors_1 = get_pert_vec_errors((pert[1], ), DE)

        true_change_vec_multi= get_pert_vec_errors(pert, DE)
        true_change_vec_0= get_pert_vec_errors((pert[0], ), DE)
        true_change_vec_1= get_pert_vec_errors((pert[1], ), DE)
        ours_change_vec = []
        for r in range(REPEAT):
            ours_change_vec.append(method_pert_mean_dicts_dict["ours"][r][pert][DE, 1] - method_pert_mean_dicts_dict["ours"][r][pert][DE, 0])

        ours_change_errors = np.array(ours_change_vec).std(0)
        ours_change_vec = np.array(ours_change_vec).mean(0)

        args = np.argsort(np.abs(ours_change_vec - true_change_vec_multi))


        width = 1/(3 + 1)
        indices = np.arange(len(the_gene_idxs))
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 4)
        ax.bar(indices + 0 * width, true_change_vec_multi[args][the_gene_idxs], label=f'True {pert_name}', width=width, color = 'gray')
        ax.bar(indices + 1 * width, true_change_vec_0[args][the_gene_idxs], label=f'True {pert_0}', width=width, color = 'orange')
        ax.bar(indices + 1 * width, true_change_vec_1[args][the_gene_idxs], label=f'True {pert_1}', width=width, bottom=true_change_vec_0[args][the_gene_idxs], color = 'blue')
        ax.bar(indices + 2 * width, ours_change_vec[args][the_gene_idxs], yerr= ours_change_errors[args][the_gene_idxs], label=f'Ours {pert_name}', width=width, color = 'green')

        ax.set_xlabel('Gene', fontsize = 17)
        ax.set_ylabel(f'Change in \n gene expression', fontsize = 15)
        ax.set_xlim(-0.5, indices[-1] + 0.75)
        plt.xticks(ticks=indices + width * (3 - 1) /2, labels=genes[args][the_gene_idxs], fontsize = 15)

        first_legend = ax.legend(loc='upper left', fontsize = 11)

        plt.tight_layout()
        plt.savefig(f'pdfs/to_show_{pert_name}.pdf', dpi = 300)
        plt.close()


################################
#Histogram for perturbations of 0/2 + 1/2 + 2/2, x: perturbations, y: r-MSE; for 2 methods (ours/GEARS)
################################

colors = {'seen 0/2': 'deeppink', 'seen 1/2': 'darkblue', 'seen 2/2': 'darkorange'} 
categories = ['seen 0/2'] * len(perts_0_2) + ['seen 1/2'] * len(perts_1_2) + ['seen 2/2'] * len(perts_2_2)
for name, scenario in scenarios.items():
    if name != "all2":
        continue
    print(f"Sceanario: {name};")
    scenario_names = np.array(['+'.join([ad.var.gene_name.tolist()[pi] for pi in pert]) for pert in scenario])

    plt.figure(figsize=(30, 5))
    indices = np.arange(len(scenario))
    width = 1 / 2
    values_errors_dict = {}

    for method_idx, method in enumerate(methods):
        values_errors_dict[method] = {}
        pert_mean_dicts = method_pert_mean_dicts_dict[method]
        rMSE_vecs = []
        for pert_mean_dict in pert_mean_dicts:
            rMSE_rows = []
            for pert in scenario:
                row = pert_mean_dict[pert]
                rMSE_rows.append(row[DE_dict[pert], :])
                #rMSE_rows.append(row)
            rMSE_rows = np.array(rMSE_rows) #Num_perts, num_DE(20), 2
            rMSE_vec = ((rMSE_rows[:, :, 0] ** 2).sum(1)) / ((rMSE_rows[:, :, 1] ** 2).sum(1)) #Num_perts,
            #print("rMSE_vec mean, max and min", rMSE_vec.mean(), rMSE_vec.max(), rMSE_vec.min())
            rMSE_vecs.append(rMSE_vec)
        rMSE_vecs = np.array(rMSE_vecs)
        values_errors_dict[method]['values'] = rMSE_vecs.mean(0) * 100
        values_errors_dict[method]['errors'] = rMSE_vecs.std(0) * 100
    
    values = values_errors_dict['ours']['values'] / values_errors_dict['gears']['values'] * 100
    errors =  values_errors_dict['ours']['errors'] / values_errors_dict['gears']['values'] * 100

    args = np.zeros_like(values, dtype = int)
    args[0: len(perts_0_2)] = np.argsort(values[0: len(perts_0_2)])
    args[len(perts_0_2): len(perts_0_2) + len(perts_1_2)] = np.argsort(values[len(perts_0_2): len(perts_0_2) + len(perts_1_2)]) + len(perts_0_2)
    args[len(perts_0_2) + len(perts_1_2): ] = np.argsort(values[len(perts_0_2) + len(perts_1_2): ]) + len(perts_0_2) + len(perts_1_2)

    fig, ax = plt.subplots()
    fig.set_size_inches(25, 5)
    ax.bar(indices, 
            values[args], 
            yerr=errors[args], 
            label='Ours', 
            width=width,
            color = 'green')
    gears_indices = np.arange(-1, len(scenario)+1)
    ax.errorbar(x=gears_indices, y=[100] * len(gears_indices), yerr=[0] * len(gears_indices), fmt="--", elinewidth=5, label = 'GEARS', color = 'red')

    ax.set_xlabel('Perturbation', fontsize = 16)
    ax.set_ylabel(f'MSE of top 20 DE genes\nrelative to GEARS (%)', fontsize = 16)
    ax.set_xlim(-1, indices[-1] + 1)
    #plt.title(f'Histogram of {column} by Model')
    plt.xticks(ticks=indices, labels=scenario_names[args], rotation=90, fontsize = 13)

    first_legend = ax.legend(loc='upper left', title='Methods', fontsize = 16, title_fontsize=16)
    #plt.setp(first_legend.get_title(),fontsize=16)
    # Add the first legend manually to the current Axes.
    plt.gca().add_artist(first_legend)
    
    for ticklabel, cat in zip(ax.get_xticklabels(), categories):
        ticklabel.set_color(colors[cat])

    # Create a second legend for categories
    category_patches = [mpatches.Patch(color=color, label=cat) for cat, color in colors.items()]
    second_legend = plt.legend(handles=category_patches, loc='upper right', bbox_to_anchor=(0.006, -0.08), title = 'Scenarios', fontsize = 16, title_fontsize=16)
    #plt.setp(second_legend.get_title(),fontsize=16)
    plt.tight_layout()
    plt.savefig(f'pdfs/pert_hist_relative_{name}.pdf', dpi = 300)
    plt.close()


################
# generate clustermap_info
################

for name, scenario in scenarios.items():
    # Example data
    #colors = {'seen 0/2': 'red', 'seen 1/2': 'blue', 'seen 2/2': 'green'} 
    #categories = ['seen 0/2'] * len(perts_0_2) + ['seen 1/2'] * len(perts_1_2) + ['seen 2/2'] * len(perts_2_2)

    DE_union = np.array([], dtype=int)
    for pert in scenario:
        DE_union = np.union1d(DE_union, DE_dict[pert])
    MSE_vecs = []
    rMSE_vecs = []
    pert_mean_dicts = method_pert_mean_dicts_dict["ours"]
    for pert_mean_dict in pert_mean_dicts:
        rMSE_rows = []
        for pert in scenario:
            row = pert_mean_dict[pert]
            rMSE_rows.append(row)
        rMSE_rows = np.array(rMSE_rows) #Num_perts, N, 2
        rMSE_rows = rMSE_rows[:, DE_union, :]
        MSE_vec = (rMSE_rows[:, :, 0] ** 2) #Num_perts, DE,
        rMSE_vec = ((rMSE_rows[:, :, 0] ** 2)) / ((rMSE_rows[:, :, 1] ** 2)) #Num_perts, DE,
        print("rMSE_vec: ", rMSE_vec.shape)
        MSE_vecs.append(MSE_vec)
        rMSE_vecs.append(rMSE_vec)
    MSE_vecs = np.array(MSE_vecs) #5, Num_perts, DE,
    rMSE_vecs = np.array(rMSE_vecs) #5, Num_perts, DE,
    MSE_vec = MSE_vecs.mean(0) #Num_perts, DE,
    rMSE_vec = rMSE_vecs.mean(0) #Num_perts, DE,
    if name == "all2":
        MSE_mat = deepcopy(MSE_vec)
        rel_MSE_mat = deepcopy(rMSE_vec)
    #rMSE_vec[rMSE_vec > 1] = 1
    rMSE_vec = np.tanh(rMSE_vec) * 100
    print("rMSE_vec: ", rMSE_vec.shape)

    y_labels = np.array(['+'.join([ad.var.gene_name.tolist()[pi] for pi in pert]) for pert in scenario])
    x_labels = np.array(ad.var.gene_name.tolist())[DE_union]  # Replace with your y labels
    if name == "all2":
        pert_names = y_labels
        de_gene_names = x_labels
        categories = np.array(['seen 0/2'] * len(perts_0_2) + ['seen 1/2'] * len(perts_1_2) + ['seen 2/2'] * len(perts_2_2))
        np.save("clustermap_info/MSE_mat_79x304.npy", MSE_mat)
        np.save("clustermap_info/rel_MSE_mat_79x304.npy", rel_MSE_mat)
        np.save("clustermap_info/pert_names_79.npy", pert_names)
        np.save("clustermap_info/pert_categories_79.npy", categories)
        np.save("clustermap_info/de_gene_names_304.npy", de_gene_names)


################
# create non-additive figures for seen_2_2.
################

for name, scenario in scenarios.items():
    if name == "all2" or name == "seen_0_1":
        continue
    print(f"Sceanario: {name};")
    if not os.path.exists(f'non_add_figs/{name}'):
        os.mkdir(f'non_add_figs/{name}')
    for pert in scenario:
        DE = DE_dict[pert]
        genes = [ad.var.gene_name.tolist()[i] for i in DE]
        pert_0 = ad.var.gene_name.tolist()[pert[0]]
        pert_1 = ad.var.gene_name.tolist()[pert[1]]
        pert_name = f"{pert_0}+{pert_1}"
        
        #true_change_vec_multi = method_pert_mean_dicts_dict["ours"][0][pert][DE, 1]
        #true_change_vec_0 = method_pert_mean_dicts_dict["ours"][0][(pert[0],)][DE, 1]
        #true_change_vec_1 = method_pert_mean_dicts_dict["ours"][0][(pert[1],)][DE, 1]
        #true_change_vec_multi, true_change_errors_multi = get_pert_vec_errors(pert, DE)
        #true_change_vec_0, true_change_errors_0 = get_pert_vec_errors((pert[0], ), DE)
        #true_change_vec_1, true_change_errors_1 = get_pert_vec_errors((pert[1], ), DE)

        true_change_vec_multi= get_pert_vec_errors(pert, DE)
        true_change_vec_0= get_pert_vec_errors((pert[0], ), DE)
        true_change_vec_1= get_pert_vec_errors((pert[1], ), DE)
        ours_change_vec = []
        gears_change_vec = []
        for r in range(REPEAT):
            ours_change_vec.append(method_pert_mean_dicts_dict["ours"][r][pert][DE, 1] - method_pert_mean_dicts_dict["ours"][r][pert][DE, 0])
            gears_change_vec.append(method_pert_mean_dicts_dict["gears"][r][pert][DE, 1] - method_pert_mean_dicts_dict["gears"][r][pert][DE, 0])

        ours_change_errors = np.array(ours_change_vec).std(0)
        ours_change_vec = np.array(ours_change_vec).mean(0)

        gears_change_errors = np.array(gears_change_vec).std(0)
        gears_change_vec = np.array(gears_change_vec).mean(0)

        width = 1/(4 + 1)
        indices = np.arange(len(DE))
        fig, ax = plt.subplots()
        fig.set_size_inches(25, 5)
        ax.bar(indices + 0 * width, true_change_vec_multi, label=f'True {pert_name}', width=width, color = 'gray')
        ax.bar(indices + 1 * width, true_change_vec_0, label=f'True {pert_0}', width=width, color = 'orange')
        ax.bar(indices + 1 * width, true_change_vec_1, label=f'True {pert_1}', width=width, bottom=true_change_vec_0, color = 'chocolate')
        ax.bar(indices + 2 * width, gears_change_vec, yerr= gears_change_errors, label=f'GEARS {pert_name}', width=width, color = 'red')
        ax.bar(indices + 3 * width, ours_change_vec, yerr= ours_change_errors, label=f'Ours {pert_name}', width=width, color = 'green')

        ax.set_xlabel('Gene', fontsize = 16)
        ax.set_ylabel(f'Change in gene expression', fontsize = 16)
        ax.set_xlim(-1, indices[-1] + 1)
        plt.xticks(ticks=indices + width * (4 - 1) /2, labels=genes, fontsize = 13) #rotation=90, 

        first_legend = ax.legend(loc='upper right', fontsize = 13)

        plt.tight_layout()
        plt.savefig(f'non_add_figs/{name}/both_methods_{pert_name}.pdf')
        plt.close()




################
# create x = gene-perturbation degree, y = r-MSE scatter plot
################


#Scatter plot for perturbations of 0/2, 1/2, 2/2, x: gene degree, y: average r-MSE
for name, scenario in scenarios.items():
    if name != "all":
        continue
    degree_vec = []
    for pert in scenario:
        for de_gene in np.union1d(DE_dict[pert], pert_graph_dict[pert]):
            degree = 0
            for gi in pert:
                gsymbol = ad.var.gene_name.tolist()[gi]
                de_gsymbol = ad.var.gene_name.tolist()[de_gene]
                ws = [go_graph[go_graph["source"] == gsymbol][go_graph["target"] == de_gsymbol]["importance"].tolist(),
                        go_graph[go_graph["source"] == de_gsymbol][go_graph["target"] == gsymbol]["importance"].tolist()]
                for w in ws:
                    if len(w) >= 1:
                        degree += w[0]
            degree_vec.append(degree)
    degree_vec = np.array(degree_vec) #DE * Num_perts,
    print(f"Sceanario: {name}; degree_vec: {len(degree_vec)}")

    for method_idx, method in enumerate(["ours"]):
        pert_mean_dicts = method_pert_mean_dicts_dict[method]
        rMSE_vecs = []
        for pert in scenario:
            for de_gene in np.union1d(DE_dict[pert], pert_graph_dict[pert]):
                rMSE_vec = []
                for pert_mean_dict in pert_mean_dicts:
                    rMSE_vec.append((pert_mean_dict[pert][de_gene, 0] ** 2) / (pert_mean_dict[pert][de_gene, 1] ** 2))
                rMSE_vecs.append(rMSE_vec)
        rMSE_vec = np.array(rMSE_vecs).mean(1) #DE * Num_perts,5 -> DE * Num_perts,
        #rMSE_vec = np.log1p(rMSE_vec)

        #args = np.where(degree_vec > 0)[0]

        #degree_vec = degree_vec[args]
        #rMSE_vec = rMSE_vec[args]

        args = np.where(rMSE_vec < 1)[0]
        degree_vec = degree_vec[args]
        rMSE_vec = rMSE_vec[args]

        slope, intercept, r_value, p_value, std_err = stats.linregress(degree_vec, rMSE_vec)

        # Create scatter plot
        plt.figure(i, figsize=(6, 4))
        plt.scatter(degree_vec, rMSE_vec, label='Data Points', marker = ",", s = np.ones_like(degree_vec))
        plt.plot(degree_vec, slope * degree_vec + intercept, color='red', label=f'Fit Line: y={slope:.2f}x+{intercept:.2f}')

        # Add regression info
        plt.text(0.05, 1.18, f'R-squared: {r_value**2:.3f}\nP-value: {p_value:.2e}\nStd Error: {std_err:.3f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        # Add legend, title, and labels
        plt.legend(loc = 'lower right', bbox_to_anchor=(1, 1))
        plt.xlabel('Perturbation-gene conectivity in GO graph') 
        plt.ylabel(f'Relative Error') #Log-transformed 
        plt.tight_layout()
        plt.savefig(f'detail_figs/pert_gene_degree_{method}_{name}.pdf')
        plt.close()