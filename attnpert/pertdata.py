from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from .data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from .utils import print_sys, zip_data_download_wrapper, filter_pert_cond_in_go

class PertData:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.gene_names = None
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            
    def load(self, data_name = None, 
             data_path = None, exclude_pert = set()): #exclude_pert = set(): Added by Ding, to make new splits.
        if data_name not in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential'] and data_path is None:
            data_path = os.path.join(self.data_path, data_name)

        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)            
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data is either Norman/replogle_k562_essential/replogle_rpe1_essential or a path to an h5ad file")
        
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        print_sys(exclude_pert)
        if len(exclude_pert) > 0:
            filter_go = self.adata.obs[self.adata.obs.condition.apply(
                                lambda x: filter_pert_cond_in_go(x, exclude_pert))]
            self.adata = self.adata[filter_go.index.values, :]

        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = list(self.adata.var.gene_name)
            conditions = list(self.adata.obs.condition.unique())
            pert_genes = []
            for condition in conditions:
                if condition == 'ctrl':
                    continue
                cond1 = condition.split('+')[0]
                cond2 = condition.split('+')[1]
                if cond1 != 'ctrl': 
                    if cond1 not in pert_genes:
                        pert_genes.append(cond1)
                    if cond1 not in self.gene_names:
                        self.gene_names.append(cond1)
                if cond2 != 'ctrl':
                    if cond2 not in pert_genes:
                        pert_genes.append(cond2)
                    if cond2 not in self.gene_names:
                        self.gene_names.append(cond2)
            self.gene_names = np.array(self.gene_names)
            self.real_gene_names = np.array(list(self.adata.var.gene_name))
            print_sys(f"len(self.gene_names): {len(self.gene_names)}")
            print_sys(f"len(self.real_gene_names): {len(self.real_gene_names)}")
            print_sys("Creating pyg object for each cell in the data...")
            self.dataset_processed = self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
            
    def new_data_process(self, dataset_name,
                         adata = None):
        
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = get_DE_genes(adata)
        self.adata = get_dropout_non_zero_genes(self.adata)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
    
        self.gene_names = list(self.adata.var.gene_name)
        conditions = list(self.adata.obs.condition.unique())
        pert_genes = []
        for condition in conditions:
            if condition == 'ctrl':
                continue
            cond1 = condition.split('+')[0]
            cond2 = condition.split('+')[1]
            if cond1 != 'ctrl': 
                if cond1 not in pert_genes:
                    pert_genes.append(cond1)
                if cond1 not in self.gene_names:
                    self.gene_names.append(cond1)
            if cond2 != 'ctrl':
                if cond2 not in pert_genes:
                    pert_genes.append(cond2)
                if cond2 not in self.gene_names:
                    self.gene_names.append(cond2)
        self.gene_names = np.array(self.gene_names)
        self.real_gene_names = np.array(list(self.adata.var.gene_name))


        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        print_sys("Creating pyg object for each cell in the data...")
        self.dataset_processed = self.create_dataset_file()
        print_sys("Saving new dataset pyg object at " + dataset_fname) 
        pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
        print_sys("Done!")
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None):
        available_splits = ['simulation', 'simulation_single', 'combo_seen0', 'combo_seen1', 
                            'combo_seen2', 'single', 'no_test', 'no_split']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.train_gene_set_size = train_gene_set_size
        
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' + str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)
            
            elif split == 'single':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction, seed=seed)
            
            elif split == 'no_test':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction, seed=seed)
            
            elif split == 'no_split':          
                adata = self.adata
                adata.obs['split'] = 'test'
            
            set2conditions = dict(adata.obs.groupby('split').agg({'condition': lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        if test_batch_size is None:
            test_batch_size = batch_size
        if self.gene_names is None:
            self.gene_names = list(self.adata.var.gene_name)
            conditions = list(self.adata.obs.condition.unique())
            pert_genes = []
            for condition in conditions:
                if condition == 'ctrl':
                    continue
                cond1 = condition.split('+')[0]
                cond2 = condition.split('+')[1]
                if cond1 != 'ctrl': 
                    if cond1 not in pert_genes:
                        pert_genes.append(cond1)
                    if cond1 not in self.gene_names:
                        self.gene_names.append(cond1)
                if cond2 != 'ctrl':
                    if cond2 not in pert_genes:
                        pert_genes.append(cond2)
                    if cond2 not in self.gene_names:
                        self.gene_names.append(cond2)
            self.gene_names = np.array(self.gene_names)
            self.real_gene_names = np.array(list(self.adata.var.gene_name))

        self.node_map = {x: it for it, x in enumerate(self.gene_names)}
        if (self.gene_names[: len(self.real_gene_names)] != self.real_gene_names).any():
            raise ValueError("self.real_gene_names must be in the start of self.gene_names")
        
        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")
        del self.dataset_processed # clean up some memory
    
        
    def create_dataset_file(self):
        dl = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            cell_graph_dataset = self.create_cell_graph_dataset(self.adata, p, num_samples=1)
            dl[p] = cell_graph_dataset
        return dl
    
    def get_pert_idx(self, pert_category, adata_):
        pert_idx = [np.where(p == self.gene_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        return pert_idx

    # Set up feature matrix and output
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        pert_feats = np.zeros(len(X[0]))
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = 1
        pert_feats = np.expand_dims(pert_feats, 0)
        feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T

        return Data(x=feature_mat, edge_index=None, edge_attr=None,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """

        num_de_genes = 20
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        de_genes = adata_.uns['rank_genes_groups_cov_all']
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category, adata_)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            padded_X = sp.lil_matrix((adata_.X.shape[0], len(self.gene_names)), dtype=adata_.X.dtype)
            padded_X[:, :len(self.real_gene_names)] = adata_.X

            for cell_z in padded_X:
                # Use samples from control as basal expression
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                padded_ctrl_X = sp.lil_matrix((ctrl_samples.X.shape[0], len(self.gene_names)), dtype=ctrl_samples.X.dtype)
                padded_ctrl_X[:, :len(self.real_gene_names)] = ctrl_samples.X
                for c in padded_ctrl_X:
                    Xs.append(c)
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            padded_X = sp.lil_matrix((adata_.X.shape[0], len(self.gene_names)), dtype=adata_.X.dtype)
            padded_X[:, :len(self.real_gene_names)] = adata_.X
            for cell_z in padded_X:
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs