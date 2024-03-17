import pandas as pd
import numpy as np
from copy import deepcopy
import os
import pickle
from tqdm import tqdm
from attnpert import PertData
from copy import deepcopy
from attnpert.utils import print_sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", type=str, default='norman')
args = parser.parse_args()
DATASET_NAME = args.dataset_name

SAVE_DIR = "data"

###############
#generate gene2vec.
###############

with open(f"{SAVE_DIR}/gene2vec_dim_200_iter_9_w2v.txt", "r") as f:
    lines = list(f.readlines())
ls_lines = [line.rstrip("\n").split(" ") for line in lines]
ls_lines.pop(0)
gene2vec_dict = {}
all_vecs = []
for ls in ls_lines:
    vec = [float(lst) for lst in ls[1:-1]]
    vec = np.array(vec)
    gene2vec_dict[ls[0]] = vec
    all_vecs.append(vec)
all_vecs = np.array(all_vecs)
print_sys(f"all_vecs: {all_vecs.shape}")
mean_vec = all_vecs.mean(axis = 0)
cov_vec = np.cov(all_vecs.T)
print_sys(f"mean_vec: {mean_vec.shape}")
print_sys(f"cov_vec: {cov_vec.shape}")

def Symbol2vec(symbol):
    try:
        vec = gene2vec_dict[symbol]
        keyerror = False
    except KeyError:
        vec = None
        keyerror = True
    return vec, keyerror

def symbols_to_vecs(symbols):
    human_symbol_vecs = {}
    cnt_no_vec = 0
    for symbol in tqdm(symbols):
        #path0: symbol -> vec
        vec, keyerror = Symbol2vec(symbol)
        if not keyerror:
            human_symbol_vecs[symbol] = vec
            continue
        #path1: vec not found. random sample.
        if symbol not in human_symbol_vecs.keys():
            human_symbol_vecs[symbol] = np.zeros(mean_vec.shape, dtype = np.float)
            cnt_no_vec += 1
    print_sys(f"number of symbol ids having no Gene2Vec: {cnt_no_vec}")
    return human_symbol_vecs

def genes_to_vec(name, symbols):
    cnt_no_vec = 0
    symbol_vecs = symbols_to_vecs(symbols)
    for ig, symbol in enumerate(symbols):
        if np.linalg.norm(symbol_vecs[symbol]) == 0:
            cnt_no_vec += 1
            symbol_vecs[symbol] = np.random.multivariate_normal(mean_vec, cov_vec)
    print_sys(f"REAL number of GENES having no VEC term: {cnt_no_vec}")
    gene2vec_npy = np.zeros((len(symbols), 200))
    for i, gene in enumerate(symbols):
        gene2vec_npy[i, :] = symbol_vecs[gene]
    print_sys(f"gene2vec_npy: {gene2vec_npy.shape}")
    if not os.path.exists(f"{SAVE_DIR}/{name}"):
        os.mkdir(f"{SAVE_DIR}/{name}")
    np.save(f"{SAVE_DIR}/{name}/gene2vec.npy", gene2vec_npy)

print_sys(f">> Dataset: {DATASET_NAME} <<")
# get data
pert_data = PertData(f'./{SAVE_DIR}')
pert_data.load(data_name = DATASET_NAME)

print_sys(f"Gene num: {pert_data.adata.shape[1]}")
genes_to_vec(DATASET_NAME, list(pert_data.gene_names))
