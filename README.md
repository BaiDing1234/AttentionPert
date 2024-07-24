# AttentionPert: Accurately Modeling Multiplexed Genetic Perturbations with Multi-scale Effects

This repository hosts the official implementation of AttentionPert, as well as the reproducing scripts of figures and tables in the paper and supplementray. 


<p align="center"><img src="https://github.com/BaiDing1234/AttentionPert/blob/main/img/model.png" alt="attnpert" width="900px" /></p>

## Installation 

```
conda env create -f environment.yml
```

If not working: Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and then do `pip install cell-gears`.

## Datasets 

### Download datasets

For the 3 datasets: "Norman", "RPE1" and "K562" used in our paper, you can download the preprocessed datasets from GEARS, and reduced GO graphs and gene2vec matrices from us.

Let's take Norman dataset as an example in following steps. 

- Run `python pertdata_example.py --dataset_name norman` to get the norman dataset from GEARS (or download [norman](https://dataverse.harvard.edu/api/access/datafile/6154020) directly and unzip it into 'data' directory).

- Remove /data/norman/go.csv (we will use reduced GO graphs).

- Download contents of "/data/norman" from [Data&Results](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EuIsFdWM1WtKqdt-NnMkwjMBAeH4bA41mghaY5Zz6LToKA?e=fL9U58)

- Move them to /data/norman. Now Norman dataset is prepared! 

For other 2 datasets, just use "replogle_rpe1_essential" or "replogle_k562_essential" to replace "norman" in these steps.

### Prepare your own datasets

For other datasets, you can also make it prepared by following steps. 

Let's suppose dataset name is "sample_data"

- See [GEARS_Data_Tutorial](https://github.com/snap-stanford/GEARS/blob/master/demo/data_tutorial.ipynb) to prepare a perturb_processed.h5ad. 

- Remove all other files, leave only perturb_processed.h5ad and move it to /data/sample_data

- Get gene2vec_dim_200_iter_9_w2v.txt from [Gene2Vec](https://github.com/jingcheng-du/Gene2vec/tree/master/pre_trained_emb) and move it to /data

- Download "/data/gene2go_all.pkl" from [Data&Results](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EuIsFdWM1WtKqdt-NnMkwjMBAeH4bA41mghaY5Zz6LToKA?e=fL9U58), move it to /data.

- Run `python gene2vec_example.py --dataset_name sample_data` to get the gene2vec matrix.

- Now it's done! 

(Though you don't see GO graph of the sample_data, don't worry. It will be automatically produced using the gene2go_all.pkl when you run the experiment for the first time.)

## Run an experiment

After you download existing dataset or prepare your own, now you can run an experiment using the script.

```
python run_attnpert.py \ 
--split 1 \ #data split seed. Could be any integer.
--repeat 5 \ #repeat times for each experiment.
--epochs 20 \ #number of epochs.
--batch_size 128 \ #batch size.
--valid_every 1 \ #number of training epochs between twice validation.
--dataset_name norman \ #dataset name, make it consistent with the directory name in /data.
-record_pred \ #store true if you want to record all the testing predictions for more detailed analysis. 
> res/attnpert_norman_log.txt 2>&1 \ #output training log and test results.
```

## Reproduce figures

Check /result_process, there is another README file for this.

## Cite Us

AttentionPert is now published!

```
@article{10.1093/bioinformatics/btae244,
    author = {Bai, Ding and Ellington, Caleb N and Mo, Shentong and Song, Le and Xing, Eric P},
    title = "{AttentionPert: accurately modeling multiplexed genetic perturbations with multi-scale effects}",
    journal = {Bioinformatics},
    volume = {40},
    number = {Supplement_1},
    pages = {i453-i461},
    year = {2024},
    month = {06},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae244},
    url = {https://doi.org/10.1093/bioinformatics/btae244},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/40/Supplement\_1/i453/58355083/btae244.pdf},
}
```
Paper: [Link](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i453/7700899)


## Acknowledgments

This project makes use of content and code from [GEARS](https://github.com/snap-stanford/GEARS), which has been instrumental in the development of this project. We are deeply grateful to the original authors and contributors for their work.


