from copy import deepcopy
import argparse
from time import time
import sys, os
import pickle

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from .model import *
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis, compute_synergy_loss, evaluate_with_nonzero_mask
from .utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params, get_go_auto

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class ATTNPERT_RECORD_TRAIN:
    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'None', 
                 exp_name = 'None'):
        
        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name, entity = "your_entity")  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = list(pert_data.gene_names)
        self.num_genes = len(self.gene_list)
        try:
            self.real_gene_list = list(pert_data.real_gene_names)
        except AttributeError:
            self.real_gene_list = list(pert_data.gene_names)
        self.real_num_genes = len(self.real_gene_list)
        if self.num_genes != self.real_num_genes:
            print_sys("The gene list is augmented, larger than real gene list!")
        self.ctrl_expression = torch.tensor(np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis = 0)).reshape(-1,).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i,j in 
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        #For loss analyzation: 
        self.train_losses = []
        self.train_overall_mses = []
        self.val_overall_mses = []
        self.train_de_mses = []
        self.val_de_mses = []
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1,
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                        model_class = PL_PW_non_add_Model,
                        **kwargs
                         ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                      }
        self.config.update(kwargs)
        
        if ('de_gene_embedding_setting' in self.config.keys()) and self.config['de_gene_embedding_setting']['use_de_emb']:
            self.de_gene_tokens = torch.zeros((self.num_genes, )).long().to(self.device)
            self.preprocess_trainloader()
            self.config['de_gene_tokens'] = self.de_gene_tokens

        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type = 'go', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_go_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions)
            print_sys(f"The go_edge_list shape: {edge_list.shape}")
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
        
        if self.config.get('shuffle_edges', False):
            self.random_shuffle_edges()

        if self.config.get('shuffle_edge_weights', False):
            self.random_shuffle_edge_weights()
            
        self.model = model_class(self.config).to(self.device)
        self.best_model = deepcopy(self.model)

    def preprocess_trainloader(self):
        train_loader = self.dataloader['train_loader']
        print_sys("Preprocessing the train set...")
        Y = []
        for step, batch in enumerate(train_loader):
            y = batch.y #B, N
            #x_ctrl = batch.x.reshape(y.shape)
            #Y.append(y - x_ctrl)
            Y.append(y)
        Y = torch.cat(Y, axis = 0).to(self.device)
        #Generate de_gene_tokens.
        if ('de_gene_embedding_setting' in self.config.keys()) and self.config['de_gene_embedding_setting']['use_de_emb']:
            Y_var_argsort = torch.argsort(Y.var(0))
            de_gene_idx = Y_var_argsort[-self.config['de_gene_embedding_setting']['de_gene_num']:]
            if self.config['de_gene_embedding_setting']['de_gene_token_type'] == 'sort':
                self.de_gene_tokens[de_gene_idx] = torch.arange(1, self.config['de_gene_embedding_setting']['de_gene_token_type']+1).to(self.device)
            elif self.config['de_gene_embedding_setting']['de_gene_token_type'] == 'one':
                self.de_gene_tokens[de_gene_idx] = 1
    
    def random_shuffle_edges(self):
        rng = np.random.default_rng()
        edge_index = np.zeros((2, self.num_genes * 20),dtype = np.int64)
        edge_weight = rng.uniform(0, 1, size = (self.num_genes * 20, ))
        nodes = np.arange(self.num_genes,dtype = np.int64)
        for i in range(self.num_genes):
            edge_index[0, i * 20: (i+1) * 20] += i
            edge_index[1, i * 20: (i+1) * 20] = rng.choice(nodes, size=(20,), replace=False)
        self.config['G_go'] = torch.tensor(edge_index).to(torch.long)
        self.config['G_go_weight'] = torch.tensor(edge_weight).to(torch.float32)
    
    def random_shuffle_edge_weights(self):
        rng = np.random.default_rng()
        edge_weight = rng.uniform(0, 1, size = self.config['G_go_weight'].shape)
        self.config['G_go_weight'] = torch.tensor(edge_weight).to(torch.float32)
    
    def train(self, epochs = 20, 
              valid_every = 1,
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')
        best_model = None
        for epoch in range(epochs):
            epoch_losses = []
            self.model.train()
            train_epoch_start = time()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch, is_train = True)
                    loss = uncertainty_loss_fct(pred[:, :self.real_num_genes], logvar, y[:, :self.real_num_genes], batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch, is_train = True)
                    loss = loss_fct(pred[:, :self.real_num_genes], y[:, :self.real_num_genes], batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})
                epoch_losses.append(loss.item())

            scheduler.step()

            epoch_loss = torch.tensor(epoch_losses).mean()
            self.train_losses.append(epoch_loss)

            train_epoch_end = time()
            train_epoch_time = (train_epoch_end - train_epoch_start)/60.0

            print_sys(f'    ==  Epoch: {epoch + 1} | Training Loss: {epoch_loss:.6f} | Time in mins: {train_epoch_time:.2f}')

            if (epoch + 1) % valid_every == 0:

                # Evaluate model performance on train and val set
                train_res = evaluate(train_loader, self.model, self.config['uncertainty'], self.device, real_num_genes=self.real_num_genes)
                val_res = evaluate(val_loader, self.model, self.config['uncertainty'], self.device, real_num_genes=self.real_num_genes)
                train_metrics, _ = compute_metrics(train_res)
                val_metrics, _ = compute_metrics(val_res)

                # Print epoch performance
                log = "Epoch {}: Train Overall MSE: {:.4f} " \
                    "Validation Overall MSE: {:.4f}. "
                print_sys(log.format(epoch + 1, train_metrics['mse'], 
                                val_metrics['mse']))
                
                # Print epoch performance for DE genes
                log = "Train Top 20 DE MSE: {:.4f} " \
                    "Validation Top 20 DE MSE: {:.4f}. "
                print_sys(log.format(train_metrics['mse_de'],
                                val_metrics['mse_de']))
                
                if self.wandb:
                    metrics = ['mse', 'pearson']
                    for m in metrics:
                        self.wandb.log({'train_' + m: train_metrics[m],
                                'val_'+m: val_metrics[m],
                                'train_de_' + m: train_metrics[m + '_de'],
                                'val_de_'+m: val_metrics[m + '_de']})
                
                if val_metrics['mse_de'] < min_val:
                    min_val = val_metrics['mse_de']
                    best_model = deepcopy(self.model)

                #For loss analyzation:
                self.train_overall_mses.append(train_metrics['mse'])
                self.val_overall_mses.append(val_metrics['mse'])
                self.train_de_mses.append(train_metrics['mse_de'])
                self.val_de_mses.append(val_metrics['mse_de'])

        self.train_losses = torch.tensor(self.train_losses)
        self.train_overall_mses = torch.tensor(self.train_overall_mses)
        self.val_overall_mses = torch.tensor(self.val_overall_mses)
        self.train_de_mses = torch.tensor(self.train_de_mses)
        self.val_de_mses = torch.tensor(self.val_de_mses)

        print_sys("Done!")
        if best_model != None:
            self.best_model = best_model
        else:
            self.best_model = deepcopy(self.model)
        if self.config.get('record_pred', False):
            self.best_model.args['start_record'] = True
        self.model = 0

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model, self.config['uncertainty'], self.device, real_num_genes=self.real_num_genes)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in out[pert].keys():
                        if m in subgroup_analysis[name].keys():
                            subgroup_analysis[name][m].append(out[pert][m])
                        else:
                            subgroup_analysis[name][m] = [out[pert][m]]

                    for m in out_non_dropout[pert].keys():
                        if m in subgroup_analysis[name].keys():
                            subgroup_analysis[name][m].append(out_non_dropout[pert][m])
                        else:
                            subgroup_analysis[name][m] = [out_non_dropout[pert][m]]

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')