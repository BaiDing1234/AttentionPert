#set +o noclobber
#python pert_gene_cluster.py > txt/pert_gene_cluster.txt 2>&1

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams.update({'font.size': 20})
import anndata
from scipy import stats

gene_names = np.load('clustermap_info/de_gene_names_304.npy')
pert_groups = np.load('clustermap_info/pert_categories_79.npy')
pert_names = np.load('clustermap_info/pert_names_79.npy')
pert_names = np.array(['+'.join(sorted([name.split('+')[1], name.split('+')[0]])) for name in pert_names])
MSE_mat = np.load('clustermap_info/MSE_mat_79x304.npy')
#MSE_mat = np.sqrt(MSE_mat)
thres = MSE_mat.mean() + 3 * MSE_mat.std()

adata = anndata.read_h5ad('../data/norman/perturb_processed.h5ad')
data = adata.X.toarray()
ctrl_mean = adata[adata.obs.condition=='ctrl'].X.toarray().mean(0)
print(ctrl_mean.shape)
data = data - ctrl_mean.reshape((1, -1))

indices = np.array(['+'.join(sorted([name.split('+')[1], name.split('+')[0]])) if '+' in name else name for name in adata.obs.condition.values])
expr_df = pd.DataFrame(data=data, index=indices, columns=adata.var.gene_name.values)


if not os.path.exists(f'fig_{thres:.3f}'):
    os.mkdir(f'fig_{thres:.3f}')
MSE_mat = np.load('clustermap_info/MSE_mat_79x304.npy')
#MSE_mat = np.sqrt(MSE_mat)
df = pd.DataFrame(data=MSE_mat, index=pert_names, columns=gene_names)


multipert_expr_df = expr_df.loc[pert_names]
multipert_expr_df['pert'] = multipert_expr_df.index

input_df = multipert_expr_df.groupby(['pert']).mean().abs()[gene_names].reset_index()
input_df.index = input_df['pert'].values
input_df = input_df.drop(columns=['pert'])
input_df = input_df.loc[df.index]
input_df = input_df[df.columns]

plot_rows = []
for col in input_df.columns:
    for row in input_df.index:
        plot_rows.append([col, row, input_df.loc[row][col], df.loc[row][col]])
plot_df = pd.DataFrame(data=plot_rows, columns=['Gene', 'Perturbation', 'True Expression Change', 'Residual Error'])


def trend_plot(df, title=None, savename=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['True Expression Change'].values, df['Residual Error'].values)
    sns.regplot(data=df, x='True Expression Change', y='Residual Error', ci=99, marker="x", color=".3", line_kws=dict(color="r"))
    plt.annotate("r-squared = {:.3f}".format(r_value**2), (1.02, 0.8), xycoords='axes fraction')
    plt.annotate("p-value = {:.3f}".format(p_value), (1.02, 0.7), xycoords='axes fraction')
    plt.annotate("slope = {:.3f}".format(slope), (1.02, 0.6), xycoords='axes fraction')
    plt.annotate("intercept = {:.3f}".format(intercept), (1.02, 0.5), xycoords='axes fraction')
    plt.annotate("stderr = {:.3f}".format(std_err), (1.02, 0.4), xycoords='axes fraction')
    
    if title:
        plt.title(title)
    if savename is not None:
        plt.savefig(f'fig_{thres:.3f}/{savename}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.clf()

trend_plot(plot_df, savename='residual_correlation')
trend_plot(plot_df[plot_df['Residual Error'] < 1.85], savename='residual_correlation_nooutliers')
trend_plot(plot_df[plot_df['Residual Error'] < thres], savename='residual_correlation_small')


input_df = multipert_expr_df.groupby(['pert']).std()[gene_names].reset_index()
input_df.index = input_df['pert'].values
input_df = input_df.drop(columns=['pert'])
input_df = input_df.loc[df.index]
input_df = input_df[df.columns]

plot_rows = []
for col in input_df.columns:
    for row in input_df.index:
        plot_rows.append([col, row, input_df.loc[row][col], df.loc[row][col]])
std_plot_df = pd.DataFrame(data=plot_rows, columns=['Gene', 'Perturbation', 'True Expression Change Std', 'Residual Error'])

def trend_plot_std(df, title=None, savename=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['True Expression Change Std'].values, df['Residual Error'].values)
    sns.regplot(data=df, x='True Expression Change Std', y='Residual Error', ci=99, marker="x", color=".3", line_kws=dict(color="r"))
    plt.annotate("r-squared = {:.3f}".format(r_value**2), (1.02, 0.8), xycoords='axes fraction')
    plt.annotate("p-value = {:.3f}".format(p_value), (1.02, 0.7), xycoords='axes fraction')
    plt.annotate("slope = {:.3f}".format(slope), (1.02, 0.6), xycoords='axes fraction')
    plt.annotate("intercept = {:.3f}".format(intercept), (1.02, 0.5), xycoords='axes fraction')
    plt.annotate("stderr = {:.3f}".format(std_err), (1.02, 0.4), xycoords='axes fraction')
    
    if title:
        plt.title(title)
    if savename is not None:
        plt.savefig(f'fig_{thres:.3f}/{savename}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.clf()

trend_plot_std(std_plot_df, savename='residual_correlation_std')
trend_plot_std(std_plot_df[plot_df['Residual Error'] < 1.85], savename='residual_correlation_std_nooutliers')
trend_plot_std(std_plot_df[plot_df['Residual Error'] < thres], savename='residual_correlation_std_small')

plot_df['True Expression Change Std'] = std_plot_df['True Expression Change Std']

print(plot_df[['Perturbation', 'Gene', 'Residual Error', 'True Expression Change', 'True Expression Change Std']].sort_values('Residual Error', ascending=False).iloc[:20].to_latex(index=False))


####################
### Clustermap
####################

gene_names = np.load('clustermap_info/de_gene_names_304.npy')
pert_groups = np.load('clustermap_info/pert_categories_79.npy')
pert_names = np.load('clustermap_info/pert_names_79.npy')
data = np.load('clustermap_info/MSE_mat_79x304.npy')

df = pd.DataFrame(data=data, index=pert_names, columns=gene_names)

lut = {
    'seen 0/2': 'deeppink',
    'seen 1/2': 'darkblue',
    'seen 2/2': 'darkorange',
}
row_colors = [lut[group] for group in pert_groups]
# sns.clustermap(np.log(df.T), col_colors=row_colors, cmap='coolwarm', vmin=-20, vmax=20)
# sns.set(font_scale=1.0)
b = sns.clustermap(
    df, 
    row_colors=row_colors, 
#     col_cluster=True,
#     row_cluster=True,
    colors_ratio=0.01,
#     standard_scale=0, # Cols
    cmap='Reds', 
    figsize=(12, 5), 
    dendrogram_ratio=(0.05, 0.2),
    cbar_pos=(0.02, 0.98, 0.01, 0.08),
    xticklabels=True,
    yticklabels=True, 
)
plt.rcParams['axes.titley'] = 1.2
plt.rcParams['axes.titlepad'] = 10
b.ax_heatmap.set_yticklabels(b.ax_heatmap.get_yticklabels(), size=2)
b.ax_heatmap.set_xticklabels(b.ax_heatmap.get_xticklabels(), size=2)
b.ax_heatmap.set_ylabel('Multiplexed Perturbations', labelpad=5, size=15)
b.ax_heatmap.set_xlabel('Genes', labelpad=0, size=15)

# ax_row_colors = b.ax_row_colors
# box = ax_row_colors.get_position()
# box_heatmap = b.ax_heatmap.get_position()
# box_col_dend = b.ax_col_dendrogram.get_position()

# b.ax_heatmap.set_position([box.x0, box_heatmap.y0, box_heatmap.width, box_heatmap.height])
# box_heatmap = b.ax_heatmap.get_position()
# b.ax_col_dendrogram.set_position([box_heatmap.x0, box_col_dend.y0, box_col_dend.width, box_col_dend.height])
# ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width*1.5, box.height])

# b.ax_cbar.set_title('Relative Error')

# b.ax_col_dendrogram.set_title('Errors Reveal Gene-specific Behavior')
# plt.title('Errors Reveal Gene-specific Behavior', loc='left')
plt.savefig('fig/clustermap_wide.pdf', bbox_inches='tight')

col_idx = b.dendrogram_col.reordered_ind

row_idxs = {}
for pertset in np.unique(pert_groups):
    pertset_idx = pert_groups == pertset
    df_set = df.iloc[pertset_idx]
    data = df_set.values
    indices = df_set.index
    columns = df_set.columns
    data = data[:, col_idx]
    columns = columns[col_idx]
    df_set = pd.DataFrame(data=data, columns=columns, index=indices)
    lut = {
        'seen 0/2': 'deeppink',
        'seen 1/2': 'darkblue',
        'seen 2/2': 'darkorange',
    }
    row_colors = [lut[group] for group in pert_groups[pertset_idx]]
    # sns.clustermap(np.log(df.T), col_colors=row_colors, cmap='coolwarm', vmin=-20, vmax=20)
    # sns.set(font_scale=1.0)
    b = sns.clustermap(
        df_set, 
        row_colors=row_colors, 
        col_cluster=False,
    #     row_cluster=False,
        colors_ratio=0.01,
#         standard_scale=1, # Cols
        cmap='Reds', 
        figsize=(12, 5), 
        dendrogram_ratio=(0.05, 0.2),
        cbar_pos=(0.02, 0.98, 0.01, 0.08),
        xticklabels=True,
        yticklabels=True, 
    )
    row_idxs[pertset] = b.dendrogram_row.reordered_ind
    plt.rcParams['axes.titley'] = 1.2
    plt.rcParams['axes.titlepad'] = 10
    b.ax_heatmap.set_yticklabels(b.ax_heatmap.get_yticklabels(), size=2)
    b.ax_heatmap.set_xticklabels(b.ax_heatmap.get_xticklabels(), size=2)
    b.ax_heatmap.set_ylabel('Multiplexed Perturbations', labelpad=5, size=15)
    b.ax_heatmap.set_xlabel('Genes', labelpad=0, size=15)

    # ax_row_colors = b.ax_row_colors
    # box = ax_row_colors.get_position()
    # box_heatmap = b.ax_heatmap.get_position()
    # box_col_dend = b.ax_col_dendrogram.get_position()

    # b.ax_heatmap.set_position([box.x0, box_heatmap.y0, box_heatmap.width, box_heatmap.height])
    # box_heatmap = b.ax_heatmap.get_position()
    # b.ax_col_dendrogram.set_position([box_heatmap.x0, box_col_dend.y0, box_col_dend.width, box_col_dend.height])
    # ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width*1.5, box.height])

    # b.ax_cbar.set_title('Relative Error')

    # b.ax_col_dendrogram.set_title('Errors Reveal Gene-specific Behavior')
    # plt.title('Errors Reveal Gene-specific Behavior', loc='left')
    plt.savefig(f'fig/clustermap_wide_split{pertset[:-2]}.pdf', bbox_inches='tight')
    plt.clf()

row_idx = []
for idx in row_idxs.values():
    row_idx.extend(list(np.array(idx).astype(int) + len(row_idx)))


data = df.values
indices = df.index
columns = df.columns
data = data[row_idx, :]
data = data[:, col_idx]
indices = indices[row_idx]
columns = columns[col_idx]
df_splits = pd.DataFrame(data=data, columns=columns, index=indices)

lut = {
    'seen 0/2': 'deeppink',
    'seen 1/2': 'darkblue',
    'seen 2/2': 'darkorange',
}
row_colors = [lut[group] for group in pert_groups]
# sns.clustermap(np.log(df.T), col_colors=row_colors, cmap='coolwarm', vmin=-20, vmax=20)
# sns.set(font_scale=1.0)
b = sns.clustermap(
    df_splits, 
    row_colors=row_colors, 
    col_cluster=False,
    row_cluster=False,
    colors_ratio=0.01,
#     standard_scale=1, # Cols
    cmap='Reds', 
    figsize=(12, 5), 
    dendrogram_ratio=(0.05, 0.2),
    cbar_pos=(0.02, 0.98, 0.01, 0.08),
    xticklabels=True,
    yticklabels=True, 
)
plt.rcParams['axes.titley'] = 1.2
plt.rcParams['axes.titlepad'] = 10
b.ax_heatmap.set_yticklabels(b.ax_heatmap.get_yticklabels(), size=2)
b.ax_heatmap.set_xticklabels(b.ax_heatmap.get_xticklabels(), size=2)
b.ax_heatmap.set_ylabel('Multiplexed Perturbations', labelpad=5, size=15)
b.ax_heatmap.set_xlabel('Genes', labelpad=0, size=15)

# ax_row_colors = b.ax_row_colors
# box = ax_row_colors.get_position()
# box_heatmap = b.ax_heatmap.get_position()
# box_col_dend = b.ax_col_dendrogram.get_position()

# b.ax_heatmap.set_position([box.x0, box_heatmap.y0, box_heatmap.width, box_heatmap.height])
# box_heatmap = b.ax_heatmap.get_position()
# b.ax_col_dendrogram.set_position([box_heatmap.x0, box_col_dend.y0, box_col_dend.width, box_col_dend.height])
# ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width*1.5, box.height])

# b.ax_cbar.set_title('Relative Error')

# b.ax_col_dendrogram.set_title('Errors Reveal Gene-specific Behavior')
# plt.title('Errors Reveal Gene-specific Behavior', loc='left')
plt.savefig('fig/clustermap_wide_splits.pdf', bbox_inches='tight')