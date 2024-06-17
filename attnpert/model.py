import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import SGConv
from .utils import print_sys
from copy import deepcopy
import os
import scipy

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class SGConv_batch_fix_graph(torch.nn.Module):
    def __init__(self, 
                  in_channels: int, 
                  out_channels: int, 
                  num_nodes: int,
                  edge_index: torch.Tensor,
                  edge_weights: torch.Tensor = None,
                  K: int = 1, 
                  min_weight: float = 0, 
                  bias: bool = True, 
                  **kwargs):
        super(SGConv_batch_fix_graph, self).__init__()
        self.Adjacency = torch.ones((num_nodes, num_nodes)).to(edge_index.device) * edge_weights.min() * min_weight
        self.Adjacency += torch.eye(num_nodes).to(edge_index.device) * (1 - edge_weights.min() * min_weight)
        for idx_i in range(edge_index.shape[1]):
            i = edge_index[0, idx_i]
            j = edge_index[1, idx_i]
            if i != j:
                self.Adjacency[i, j] = edge_weights[idx_i]
        print_sys(f"Num edges in SGConv_batch_fix_graph: {edge_index.shape[1]}")

        self.Degree = self.Adjacency.sum(1)  #(N, ) diagonal.
        print_sys(f"Total Degree and Min Degree: {self.Degree.sum()}, {self.Degree.min()}")
        self.norm_Adj = ((1 / torch.sqrt(self.Degree)).unsqueeze(-1) * self.Adjacency) * (1 / torch.sqrt(self.Degree)).unsqueeze(0) #D^{-1/2}AD^{-1/2}
        print_sys(f"norm_Adj sum, min, is_nan: {self.norm_Adj.sum()}, {self.norm_Adj.min()}, {torch.isnan(self.norm_Adj).any()}")
        self.sgconv_mat = deepcopy(self.norm_Adj)
        for i in range(K - 1):
            self.sgconv_mat = self.sgconv_mat @ self.norm_Adj
        print_sys(f"sgconv_mat sum, min, is_nan: {self.sgconv_mat.sum()}, {self.sgconv_mat.min()}, {torch.isnan(self.sgconv_mat).any()}")
        del self.Adjacency, self.Degree, self.norm_Adj 

        self.sgconv_mat = self.sgconv_mat.reshape((1, num_nodes, num_nodes)) #1, N, N
        self.sgconv_mat.requires_grad = False 
        self.lin = Linear(in_channels, out_channels, bias=bias) #D_in, D_out
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        #x shape: B, N, D_in or 1, N, D_in
        #self.sgconv_mat: 1, N, N
        #Output: B, N, D_out or N, D_out
        #The multiplication will be broadcast.
        return self.lin(self.sgconv_mat @ x).squeeze()



class PL_PW_non_add_Model(torch.nn.Module):
    """
    AttentionPert Model
    """

    def __init__(self, args: dict):
        super(PL_PW_non_add_Model, self).__init__()
        self.args = args       
        if 'exp_name' in args.keys():
            self.exp_name = args['exp_name']
        if args.get('record_pred', False):
            self.pred_dir = f"./result_process/preds/{self.exp_name}"
            if not os.path.exists(self.pred_dir):
                os.mkdir(self.pred_dir)
            self.pred_batch_idx = 0

        self.num_genes = args['num_genes']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']

        #self.ablation_strings = {"g2v": "N"/"Y", "app": "N"/"S"/"R" , "pl": "N"/"Y",  "pw": "N"/"S"/"R"}
        self.ablation_strings = {"g2v": "Y", "app": "R" , "pl": "Y",  "pw": "R"}

        self.gene2vec_args = args["gene2vec_args"] #{"base_use_gene2vec", "pert_local_use_gene2vec", "pert_use_gene2vec", "pert_weight_use_gene2vec"}, all True
        self.pert_local_min_weight = args['pert_local_min_weight'] #0~1, use 0 or 1
        self.pert_local_conv_K = args["pert_local_conv_K"] #1~N, use 1 or 2

        if self.ablation_strings["app"] == "R" and self.ablation_strings["pl"] == "N":
            raise ValueError("self.ablation_strings: app being R and pl being N is illegal!")        
        # perturbation positional embedding added only to the perturbed genes
           
        # gene/globel perturbation embedding dictionary lookup
        if self.ablation_strings["g2v"] == "Y":
            gene2vec_weight = np.load(self.gene2vec_args['gene2vec_file'])     
            gene2vec_weight = torch.from_numpy(gene2vec_weight).float()
            emb_hidden_dim = gene2vec_weight.shape[1]
        else:
            emb_hidden_dim = hidden_size

        
        self.non_add_beta = args['non_add_beta']
        self.non_add_mat = nn.Parameter(torch.rand(self.num_genes, emb_hidden_dim)) #N, D_e

        if self.ablation_strings["pl"] == "Y":
            if self.ablation_strings["g2v"] == "Y":
                self.pert_local_emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False)
            else:
                self.pert_local_emb = nn.Embedding(self.num_genes, emb_hidden_dim, max_norm=True)
            self.sgconv_batch = SGConv_batch_fix_graph(in_channels = emb_hidden_dim, 
                                                       out_channels = emb_hidden_dim,
                                                       num_nodes = self.num_genes,
                                                       edge_index = args['G_go'].to(args['device']),
                                                       edge_weights=args['G_go_weight'].to(args['device']),
                                                       K = self.pert_local_conv_K, 
                                                       min_weight = self.pert_local_min_weight)
            self.pert_local_fuse = MLP([emb_hidden_dim, hidden_size], last_layer_act='ReLU')
            self.pert_one_emb = nn.Linear(1, emb_hidden_dim)
        
        if self.ablation_strings["app"] == "S":   
            self.pert_w = nn.Linear(1, emb_hidden_dim)
        if self.ablation_strings["app"] != "N":
            if emb_hidden_dim != hidden_size:
                self.pert_w_mlp = nn.Linear(emb_hidden_dim, hidden_size)
            else:
                self.pert_w_mlp = lambda x: x

        if self.ablation_strings["pw"] != "N":
            if self.ablation_strings["g2v"] == "Y":
                self.pert_weight_emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False)
                self.pert_weight_single_emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False)
            else:
                self.pert_weight_emb = nn.Embedding(self.num_genes, emb_hidden_dim, max_norm=True)
                self.pert_weight_single_emb = nn.Embedding(self.num_genes, emb_hidden_dim, max_norm=True)
            
            self.pw_heads = args["pert_weight_heads"]
            self.pw_head_dim = args["pert_weight_head_dim"]
            self.pert_weight_gnn_to_v = SGConv(emb_hidden_dim, self.pw_heads * self.pw_head_dim, 1, bias = False)
            self.pert_weight_gnn_to_q = SGConv(emb_hidden_dim, self.pw_heads * self.pw_head_dim, 1, bias = False)
            self.pert_weight_gnn_to_k = SGConv(emb_hidden_dim, self.pw_heads * self.pw_head_dim, 1, bias = False)
            self.pert_weight_single_fuse = MLP([self.pw_heads * self.pw_head_dim, hidden_size], last_layer_act='ReLU')

            pert_weight_act_key = args.get("pert_weight_act", "softmax")
            if pert_weight_act_key == "softmax":
                self.pert_weight_act = lambda x: F.softmax(x / (self.pw_head_dim ** (1 / 2)), dim=2)
            elif pert_weight_act_key == "tanh":
                self.pert_weight_act = torch.tanh
            elif pert_weight_act_key == "sigmoid":
                self.pert_weight_act = torch.special.expit
            elif pert_weight_act_key == "maxnorm":
                self.pert_weight_act = lambda x: x / (torch.max(torch.abs(x), dim=2, keepdim=True).values + 1e-12)
            else:
                raise ValueError(f"input args pert_weight_act wrong!")

        if self.ablation_strings["pw"] != "R":
            if self.ablation_strings["g2v"] == "Y":
                self.pert_emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False)
            else:
                self.pert_emb = nn.Embedding(self.num_genes, emb_hidden_dim, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([emb_hidden_dim, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([emb_hidden_dim, hidden_size, hidden_size], last_layer_act='ReLU')
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(emb_hidden_dim, emb_hidden_dim, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(emb_hidden_dim)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        

    def forward(self, data, is_train = False):
        #self.ablation_strings = {"g2v": "N"/"Y", "app": "N"/"S"/"R" , "pl": "N"/"Y",  "pw": "N"/"S"/"R"}

        if str(type(data)) == "<class 'torch.Tensor'>":
            x = data.reshape((-1, 2))
        else:
            x, batch = data.x, data.batch
        x_pert = x[:, 1].reshape((-1, self.num_genes)) #B, N
        pert = x_pert.reshape(-1,1) #B * N, 1
        num_graphs = len(data.batch.unique())
        
        base_emb = 0

        ####################
        # PertLocal #
        ####################

        if self.ablation_strings["pl"] == "Y":
            ## get base gene embeddings
            pert_local_embeddings = self.pert_local_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device'])) #B * N, D_e
            pert_indicator_emb = self.pert_one_emb(pert) #B * N, D_e
            non_add_bias = 1 + self.non_add_beta * (torch.sum(x_pert, dim = 1, keepdim = True)-1) * torch.tanh(x_pert @ self.non_add_mat) #B, D_e
            pert_bias_emb = pert_indicator_emb.reshape((num_graphs, self.num_genes, -1)) * non_add_bias.reshape((num_graphs, 1, -1)) #B, N, D_e

            pert_local_embeddings = pert_local_embeddings.reshape((num_graphs, self.num_genes, -1)) #B, N, D_e
            pert_local_embeddings += pert_bias_emb
            pert_local_embeddings = self.sgconv_batch(pert_local_embeddings) #B, N, D_e
            pert_local_embeddings = pert_local_embeddings.reshape((num_graphs * self.num_genes, -1)) #B * N, D_e
            pert_local_embeddings = self.pert_local_fuse(pert_local_embeddings) #B * N, D
            pert_local_embeddings = pert_local_embeddings.reshape((num_graphs, self.num_genes, -1)) #B, N, D
            base_emb += pert_local_embeddings

        if self.ablation_strings["app"] != "N":
            ## add the perturbation positional embedding
            if self.ablation_strings["app"] == "S":
                pert_emb = self.pert_w_mlp(self.pert_w(pert)).reshape((num_graphs, self.num_genes, -1))
            elif self.ablation_strings["app"] == "R":
                pert_emb = self.pert_w_mlp(pert_indicator_emb).reshape((num_graphs, self.num_genes, -1))
            base_emb = pert_emb + base_emb

        ####################
        # PertWeight #
        ####################

        if self.ablation_strings["pw"] != "N":
            pert_weight = self.pert_weight_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))      #N, D_e
            pert_weight_q = self.pert_weight_gnn_to_q(pert_weight, self.G_sim, self.G_sim_weight) #N, heads * head_dim
            pert_weight_k = self.pert_weight_gnn_to_k(pert_weight, self.G_sim, self.G_sim_weight) #N, heads * head_dim

            pert_weight_single_embeddings_old = self.pert_weight_single_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))      #N, D_e
            pert_weight_single_embeddings = self.pert_weight_gnn_to_v(pert_weight_single_embeddings_old, self.G_sim, self.G_sim_weight) #N, heads * head_dim
            pert_weight_single_embeddings = torch.stack(list((pert_weight_single_embeddings,)) * num_graphs)   #B, N, heads * head_dim
            pert_weight_single_embeddings = pert_weight_single_embeddings * x_pert.reshape((num_graphs, self.num_genes, 1)) #B, N, heads * head_dim
            #### multi-head-attn pert weight part ###
            pert_weight_q = pert_weight_q.reshape(self.num_genes, self.pw_heads, self.pw_head_dim) #N, heads, head_dim
            pert_weight_k = pert_weight_k.reshape(self.num_genes, self.pw_heads, self.pw_head_dim) #N, heads, head_dim
            pert_weight_single_embeddings = pert_weight_single_embeddings.reshape(num_graphs, self.num_genes, self.pw_heads, self.pw_head_dim) #B, N, heads, head_dim
            attention = torch.einsum("qhd,khd->hqk", [pert_weight_q, pert_weight_k]) #heads, N, N
            attention = self.pert_weight_act(attention) #heads, N, N
            pert_weight_single_embeddings = torch.einsum("hqk,bkhd->bqhd", [attention, pert_weight_single_embeddings]).reshape(
                num_graphs, self.num_genes, self.pw_heads * self.pw_head_dim) #B, N, heads * head_dim
            ############################################
            pert_mask = (x_pert.sum(1) != 0).reshape((num_graphs, 1, 1)) #B, 1, 1
            pert_weight_single_embeddings = pert_weight_single_embeddings.reshape((num_graphs * self.num_genes, -1)) #B * N, heads * head_dim
            pert_weight_single_embeddings = self.pert_weight_single_fuse(pert_weight_single_embeddings) #B * N, D
            pert_weight_single_embeddings = pert_weight_single_embeddings.reshape((num_graphs, self.num_genes, -1)) * pert_mask #B, N, D
            base_emb += pert_weight_single_embeddings


        #Usual General Pert.
        if self.ablation_strings["pw"] == 'R':
            pert_embeddings = pert_weight_single_embeddings_old
        else:
            pert_embeddings = self.pert_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))      #N, D
        for idx, layer in enumerate(self.sim_layers):
            pert_embeddings = layer(pert_embeddings, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_embeddings = pert_embeddings.relu() 
        pert_embeddings = torch.stack(list((pert_embeddings,)) * num_graphs)   #B, N, D_e
        pert_embeddings = pert_embeddings * x_pert.reshape((num_graphs, self.num_genes, 1)) #B, N, D_e
        pert_embeddings = pert_embeddings.sum(1) #B, D_e
        pert_mask = (x_pert.sum(1) != 0).reshape((num_graphs, 1)) #B, 1
        pert_embeddings = self.pert_fuse(pert_embeddings) * pert_mask #B, D
        pert_embeddings = pert_embeddings.reshape((num_graphs, 1, pert_embeddings.shape[1])) #B, 1, D
        base_emb += pert_embeddings

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1) #B * N, D
        
        ####################
        # Decoder #
        ####################

        base_emb = self.bn_pert_base(base_emb) #B * N, D    #bn

        base_emb = self.transform(base_emb)                 #Relu
        out = self.recovery_w(base_emb) #B * N, D
        out = out.reshape(num_graphs, self.num_genes, -1) #B, N, D
        out = out.unsqueeze(-1) * self.indv_w1 #B, N, D, 1
        w = torch.sum(out, axis = 2) ##B, N, 1
        out = (w + self.indv_b1).squeeze(2) #B, N

        out = out.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1) #B * N, 1

        ####################
        # Record test predictions #
        ####################

        if (not is_train) and self.args.get('start_record', False):
            y_flat = data.y.reshape(-1, 1) #B * N, 1
            pred = torch.cat((pert, out, y_flat), dim = 1) #B * N, 3
            pred = pred.cpu().to_sparse()

            # Convert the sparse tensor to a SciPy sparse matrix (COO format)
            values = pred.values().numpy()
            indices = pred.indices().numpy()
            sparse_matrix = scipy.sparse.coo_matrix((values, indices), shape=pred.shape)

            # Save the sparse matrix in NPZ format
            scipy.sparse.save_npz(f'{self.pred_dir}/b{self.pred_batch_idx}_{num_graphs}.npz', sparse_matrix)
            self.pred_batch_idx += 1

        ####################
        # Return #
        ####################  
              
        out = torch.split(torch.flatten(out), self.num_genes) #B, N

        ## uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)

        return torch.stack(out) #B, N



