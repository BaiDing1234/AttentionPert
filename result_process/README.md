# Reproduce figures

This directory contains all scripts to generate all Figures of results and 2 Tables in our main text and supplementary.

## Download results

Download "/preds", "/csvs" and "/data/essential_all_data_pert_genes.pkl" from [Data&Results](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EuIsFdWM1WtKqdt-NnMkwjMBAeH4bA41mghaY5Zz6LToKA?e=fL9U58).

Move /preds and /csvs to this directory, and essential_all_data_pert_genes.pkl to ../data.

(You can also run experiments to generate tables and test results.)

## Generate figures:

Using the scripts, you can reproduce the figures of results in our paper.

detail_analysis.py

| Script | Figure |
|-----------------|-------------|
| [detail_analysis.py](detail_analysis.py) | Fig. 3, 6; Supplementary Fig. 9; clustermap_info for "pert_gene_cluster.py" |
| [norman_pod_psd.py](norman_pod_psd.py) | Fig. 4; Supplementary Fig. 2 |
| [rpe1_pod_psd.py](rpe1_pod_psd.py) | Supplementary Fig. 3(a) |
| [k562_pod_psd.py](k562_pod_psd.py) | Supplementary Fig. 3(b) |
| [ablation_plot.py](ablation_plot.py) | Supplementary Fig. 4 |
| [pert_gene_cluster.py](pert_gene_cluster.py) | Fig. 5; Supplementary Fig. 5, 6, 7, 8; needs clustermap_info |

Fig. 1, 2 and Supplementary Fig. 1 are generated by using the illustrator. 

Fig. 5 and Supplementary Fig. 5, 6, 7 are also modified manually to be clearly referenced.

## Generate special tables:

Table 5 in main text contains metrics "MSE(NAG)" not produced by the automatic test following each training. 

Supplementary Table 8 is also produced automatically by script.

All other tables except Table 5 and Supplementary Table 9 are produced manually. 

Here we also provide code to generate Table 5 and Supplementary Table 8.

| Script | Table |
|-----------------|-------------|
| [NA_analysis.py](NA_analysis.py) | Table 5; saves a csv file |
| [pert_gene_cluster.py](pert_gene_cluster.py) | Supplementary Table 8; needs clustermap_info; prints latex table |