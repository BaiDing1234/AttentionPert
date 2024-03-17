import sys
sys.path.append('../')
import re

from attnpert import PertData
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", type=str, default='norman')
args = parser.parse_args()
DATASET_NAME = args.dataset_name


pert_data = PertData('./data')
pert_data.load(data_name = DATASET_NAME)