import sys
sys.path.append('../')
import re

from attnpert import PertData
from attnpert.attnpert import ATTNPERT_RECORD_TRAIN
from attnpert.model import *
from attnpert.utils import print_sys
import argparse

################
# Final result for act function: 
#          == SOFTMAX ==
################

#set +o noclobber

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--repeat", type=int, default=5, help='repeat times for each experiment.')
parser.add_argument("--epochs", type=int, default=20, help='Number of epochs.')
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--set_idx", type=int, default=3)

parser.add_argument("--dataset_name", type=str, default='norman')
parser.add_argument("--act", type=str, default='softmax')
parser.add_argument("--gene2vec_file", type=str, default='data/norman/gene2vec.npy')
parser.add_argument("-test", action='store_true')
parser.add_argument("-record_pred", action='store_true')
parser.add_argument("--cont_file", type=str, default='None')
parser.add_argument("--beta", type=float, default = 5e-2)

MODEL_CLASS = PL_PW_non_add_Model
args = parser.parse_args()
seed = args.split
REPEAT = args.repeat
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
VALID_EVERY = args.valid_every
DATASET_NAME = args.dataset_name
gene2vec_file_path = args.gene2vec_file
if 'norman' in gene2vec_file_path:
    gene2vec_file_path = gene2vec_file_path.replace('norman', DATASET_NAME)
TEST = args.test
record_pred = args.record_pred
set_idx = args.set_idx
ACT = args.act
if ACT not in ["softmax", "tanh", "sigmoid", "maxnorm"]:
    raise ValueError(ACT)
beta = args.beta

default_setting = {"gene2vec_args": {"gene2vec_file": gene2vec_file_path}, 
                    "pert_local_min_weight": 0.75, 
                    "pert_local_conv_K": 1,
                    "pert_weight_heads": 2,
                    "pert_weight_head_dim": 64,
                    "pert_weight_act": ACT,
                    "non_add_beta": beta,
                    'record_pred': record_pred}


cont_file = args.cont_file
done_exps = []

if cont_file != 'None':
    exp_pattern = re.compile(r"EXPERIMENT: (.+)")
    with open(cont_file, 'r') as f:
        for line in f:
            exp_match = exp_pattern.match(line)
            if exp_match:
                experiment_name = exp_match.group(1)
                done_exps.append(experiment_name)
    del done_exps[-1]
    print_sys(f"done_exps num, {len(done_exps)}")


pert_data = PertData('./data')
pert_data.load(data_name = DATASET_NAME)
pert_data.prepare_split(split = 'simulation', seed = seed)
pert_data.get_dataloader(batch_size = BATCH_SIZE, test_batch_size = BATCH_SIZE)

for j in range(REPEAT):
    wandb = False
    exp_name = f'AP_{DATASET_NAME}_beta_{beta:.3f}_seed_{seed}_{j}'
    if exp_name in done_exps:
        continue
    print_sys("EXPERIMENT: " + exp_name)

    attnpert_model = ATTNPERT_RECORD_TRAIN(pert_data, device = 'cuda',  
                        weight_bias_track=wandb,
                        proj_name = 'attnpert',
                        exp_name = exp_name)
    attnpert_model.model_initialize(hidden_size = 64, 
                                model_class=MODEL_CLASS,
                                exp_name = exp_name, 
                                **default_setting)
    print_sys(attnpert_model.config)
    attnpert_model.train(epochs = EPOCHS, 
                    valid_every= VALID_EVERY)