 # %% Import modules
import os, copy, anndata, natsort
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq


# %% Define functions
def load_xenium(cell_feature_matrix_path, cell_csv_path, min_counts=10, min_cells=5):
    # Load raw data
    adata = sc.read_10x_h5(filename=cell_feature_matrix_path)
    df = pd.read_csv(cell_csv_path)
    df.set_index(adata.obs_names, inplace=True)
    print("Data is loaded")
    # Set obsm
    adata.obs = df.copy()
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
    # Print data quality
    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)
    cprobes = (adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100)
    cwords = (adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100)
    print(f"Negative DNA probe count % : {cprobes}")
    print(f"Negative decoding count % : {cwords}")
    adata.layers["counts"] = adata.X.copy()
    # %% Filter cell and genes
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata, cprobes, cwords

def pre_processing(adata):
    # Normalize counts per cell
    sc.pp.normalize_total(adata, inplace=True)
    # Logarithmize
    sc.pp.log1p(adata)
    # Principal component analysis
    sc.tl.pca(adata)
    # Compute a neighborhood graph
    sc.pp.neighbors(adata)
    # Embed the neighborhood graph of the data
    sc.tl.umap(adata)
    # Cluster the cells into subgroups using leiden
    sc.tl.leiden(adata)
    # Cluster the cells into subgroups using louvain
    sc.tl.louvain(adata)
    print("Processing is complete.")

# %% Load data
root = os.getcwd()
data_root = os.path.join(root, 'data')
data_list = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
natsort.natsorted(data_list)
adatas = []
df = []
for data in data_list:
    adata, cprobes, cwords = load_xenium(os.path.join(data_root, data,"cell_feature_matrix.h5"), os.path.join(data_root, data,"cells.csv.gz"))
    adatas.append(adata)
    df.append([data, list(adata.var_names), cprobes, cwords])

common_var_names = list(set(df[0][1]).intersection(df[1][1],df[2][1],df[3][1],df[4][1],df[5][1]))
print("There are {} common variables in dataset.".format(len(common_var_names)))
adatas = [adata[:,common_var_names] for adata in adatas]

for adata in adatas:
    pre_processing(adata)

# %% Integrate data
adata_combined = sc.AnnData.concatenate(adatas[0], adatas[1], adatas[2], adatas[3], adatas[4], adatas[5], batch_categories=data_list)
adata_combined.write(os.path.join(data_root, data,"combined_adata.h5ad"))
sc.pl.umap(adata_combined, color=['batch', 'louvain'], wspace=0.4)


adata_combined = sc.read(os.path.join(data_root, data,"combined_adata.h5ad"))
sc.pp.pca(adata_combined)
sc.pp.neighbors(adata_combined)
# sc.tl.tsne(adata_combined)
sc.tl.umap(adata_combined)
sc.tl.leiden(adata_combined)
sc.tl.louvain(adata_combined)
sc.pl.umap(adata_combined, color=['batch','louvain'], wspace=0.4)
adata_combined.write(os.path.join(data_root, "combined_adata_pcaumap.h5ad"))


adata_combined = sc.read(os.path.join(data_root, "combined_adata.h5ad"))
sc.pp.pca(adata_combined)
sc.pp.neighbors(adata_combined)
sc.external.pp.bbknn(adata_combined, batch_key='batch')
sc.tl.umap(adata_combined)
sc.tl.leiden(adata_combined)
sc.tl.louvain(adata_combined)
sc.pl.umap(adata_combined, color=['batch','louvain'], wspace=0.4)
adata_combined.write(os.path.join(data_root, "combined_adata_bbknn.h5ad"))


# %% Load
adata_combined = sc.read(os.path.join(data_root, "combined_adata_bbknn.h5ad"))

# Splitting observations into subgroups based on the 'batch' metadata
batches = adata_combined.obs['batch'].unique()  # Get unique group values
adatas = {}
for batch in batches:
    subgroup_mask = adata_combined.obs['batch'] == batch  # Create a boolean mask for the current group
    subgroup_data = adata_combined[subgroup_mask].copy()  # Create a copy of the data for the subgroup
    adatas[batch] = subgroup_data  # Store the subgroup data in a dictionary


# %% Plot distribution of total transcripts per cell
n_cols = 4
n_rows = len(batches)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 3, n_cols * 5))
for i, batch_id in enumerate(batches):
    # Create a mask for the current batch
    adata = adatas[batch_id]
    axes[i, 0].set_title("Total transcripts per cell")
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axes[i, 0])
    axes[i, 1].set_title("Unique transcripts per cell")
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axes[i, 1])
    axes[i, 2].set_title("Area of segmented cells")
    sns.histplot(adata.obs["cell_area"], kde=False, ax=axes[i, 2])
    axes[i, 3].set_title("Nucleus ratio")
    sns.histplot(adata.obs["nucleus_area"] / adata.obs["cell_area"], kde=False, ax=axes[i, 3])
plt.tight_layout()
plt.show()

# %% Plot examples
sc.pl.umap(adata_combined, color=['batch','leiden'], wspace=0.4)
sc.pl.umap(adatas[batches[0]], color=['batch','leiden'], wspace=0.4)
sq.pl.spatial_scatter(adatas[batches[0]], library_id="spatial", shape=None, color=["leiden"], wspace=0.4)
plt.show()

sc.tl.embedding_density(adata_combined, groupby='batch')
sc.pl.embedding_density(adata_combined, groupby='batch')

sc.pl.highest_expr_genes(adata_combined, n_top=20, show=False)
plt.tight_layout()
plt.show()

sc.pl.violin(adata_combined, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)

sc.tl.paga(adata_combined)
sc.pl.paga(adata_combined)

# Finding marker genes for each clusters
sc.tl.rank_genes_groups(adata_combined, 'leiden', method='t-test')
sc.tl.rank_genes_groups(adata_combined, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_combined, n_genes=25, sharey=False)

sc.pl.rank_genes_groups_dotplot(adata_combined, n_genes=5, show=False, figsize=(12, 6))
plt.tight_layout()
plt.show()
# Better to look
sc.pl.rank_genes_groups_dotplot(adata_combined, n_genes=5, values_to_plot='logfoldchanges', min_logfoldchange=2,
                                vmax=7, vmin=-7, cmap='bwr', show=False, figsize=(12, 6))
plt.tight_layout()
plt.show()

sc.pl.rank_genes_groups_heatmap(adata_combined, n_genes=3, use_raw=False, swap_axes=True, vmin=-3, vmax=3,
                                cmap='bwr', figsize=(10,7), show=False)
plt.tight_layout()
plt.show()

# Make pandas dataframe
groups = adata_combined.uns['rank_genes_groups']['names'].dtype.names
rank_genes_groups = pd.DataFrame({group + '_' + key[:1]: adata_combined.uns['rank_genes_groups'][key][group]
                                  for group in groups for key in ['names', 'pvals']})
rank_genes_groups.head(5)





sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata_combined)
highly_variable_genes = adata_combined[:, adata_combined.var.highly_variable].var_names

sc.pl.dotplot(adata_combined, highly_variable_genes, groupby='leiden', show=False, figsize=(12, 7))
plt.tight_layout()
plt.show()


# Plot Umap for all adata
n_cols = 3
n_rows = len(batches)
fig, axes = plt.subplots(n_rows+1, n_cols, figsize=(n_rows * 3, n_cols * 12))
sc.pl.umap(adata_combined, color=['batch'], ax=axes[0, 0], legend_fontsize=10, legend_loc='on data', show=False)
axes[0, 0].set_title('All Batches')
sc.pl.umap(adata_combined, color=['leiden'], ax=axes[0, 1], legend_fontsize=10, legend_loc='on data', show=False)
axes[0, 1].set_title('Leiden clusters of All Batches')
axes[0, 2].axis('off')
# Loop through batches and plot UMAP on different subplots
for i, batch_id in enumerate(batches):
    # Create a mask for the current batch
    batch_mask = adata_combined.obs['batch'] == batch_id
    # Plot UMAP for the current batch
    sc.pl.umap(adata_combined[batch_mask], groups = [batch_id], color= ['batch'], ax=axes[i+1,0], legend_fontsize=10, legend_loc='on data', show=False)
    axes[i+1, 0].set_title(f'Batch {batch_id}')
    sc.pl.umap(adata_combined[batch_mask], groups = [batch_id], color=['leiden'], ax=axes[i+1, 1], legend_fontsize=10, legend_loc='on data', show=False)
    axes[i+1, 1].set_title(f'Leiden clusters of Batch {batch_id}')
    sq.pl.spatial_scatter(adata_combined[batch_mask], library_id="spatial", shape=None,color= ['leiden'], ax=axes[i+1,2])
    axes[i+1, 2].set_title(f'Batch {batch_id}')
# Adjust layout and show the plot
plt.tight_layout()
plt.show()

"""
# Spatial scatter plot for all data
n_cols = 2
n_rows = len(batches)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 2, n_cols * 15))
# Loop through batches and plot UMAP on different subplots
for i, batch_id in enumerate(batches):
    # Create a mask for the current batch
    batch_mask = adata_combined.obs['batch'] == batch_id
    # spatial scatter plot for the current batch
    sq.pl.spatial_scatter(adata_combined[batch_mask], library_id="spatial", shape=None,color= ['leiden'], ax=axes[i,0])
    axes[i, 0].set_title(f'Batch {batch_id}')
    sc.pl.umap(adata_combined[batch_mask], color=['louvain'], ax=axes[i, 1], legend_fontsize=10, legend_loc='on data', show=False)
    axes[i, 1].set_title(f'Louvain clusters of Batch {batch_id}')
# Adjust layout and show the plot
plt.tight_layout()
plt.show()
"""
"""
ingest occurs a value error: all input arrays must have the same shape.
it is due to the behavior of sc.pp.neighbors. 
Cells are sometimes given different numbers of neighbors. 
Sometimes that the errant cells have a number of neighbors greater than zero
ingest automatically checks the uns['pca']['params']['zero_center'], but 'sc.pp.pca' doesn't make it. 
Only sc.tl.pca can do it! But there is an internal bug in ingest function. 
Scanpy team didn't solve the problem yet.

sc.tl.ingest(adata_2, adata_1, obs='leiden', embedding_method='umap')
adata_concat = adata_2.concatenate(adata_1, batch_categories=['ref', 'new'])
adata_concat.obs.louvain = adata_concat.obs.louvain.astype('category')
adata_concat.obs.louvain.cat.reorder_categories(adata_2.obs.louvain.cat.categories)  # fix category ordering
adata_concat.uns['louvain'] = adata_2.uns['louvain']  # fix category colors
sc.pl.umap(adata_concat, color=['batch', 'louvain'])
"""


# %% Check common_marker_genes and meta genes
df_ref_panel_ini = pd.read_excel("D:/Projects/Xenium/data/Xenium_HumanBrainPanel_GeneList_removed_double_columns.xlsx", index_col=0)
# df_ref_panel_ini = pd.read_csv('/home/ext_choi_yoonho_mayo_edu/jupyter/Reference.csv', index_col=0)
df_ref_panel = df_ref_panel_ini.iloc[1:, 1:] # Check 'Annotation' column
df_ref_panel.index.name = None
df_ref_panel.columns = ["Function"]
marker_genes = df_ref_panel["Function"].index.tolist()

from copy import deepcopy
meta_gene = deepcopy(adata.var)
common_marker_genes = list(set(meta_gene.index.tolist()).intersection(marker_genes))
meta_gene.loc[common_marker_genes, "Markers"] = df_ref_panel.loc[common_marker_genes, "Function"]
# meta_gene["Markers"] = meta_gene["Markers"].apply(lambda x: "N.A." if "marker" not in str(x) else x)
meta_gene["Markers"].value_counts()


# %% Calculate Leiden Cluster Average Expression Signatures
ser_counts = adata_combined.obs["leiden"].value_counts()
ser_counts.name = "cell counts"
meta_leiden = pd.DataFrame(ser_counts)
cat_name = "leiden"
sig_leiden = pd.DataFrame(columns=adata_combined.var_names, index=adata_combined.obs[cat_name].cat.categories)
for clust in adata_combined.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = adata_combined[adata_combined.obs[cat_name].isin([clust]), :].X.mean(0)
sig_leiden = sig_leiden.transpose()
leiden_clusters = ["Leiden-" + str(x) for x in sig_leiden.columns.tolist()]
sig_leiden.columns = leiden_clusters
meta_leiden.index = sig_leiden.columns.tolist()
meta_leiden["leiden"] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())


# %% Assign cell types for each Leiden cluster based on meta genes
meta_gene = pd.DataFrame(index=sig_leiden.index.tolist())
meta_gene["info"] = pd.Series("", index=meta_gene.index.tolist())
meta_gene["Markers"] = pd.Series("N.A.", index=sig_leiden.index.tolist())
meta_gene.loc[common_marker_genes, "Markers"] = df_ref_panel.loc[common_marker_genes, "Function"]
meta_leiden["Cell_Type"] = pd.Series("N.A.", index=meta_leiden.index.tolist())
num_top_genes = 30
for inst_cluster in sig_leiden.columns.tolist():
    top_genes = (sig_leiden[inst_cluster].sort_values(ascending=False).index.tolist()[:num_top_genes])
    inst_ser = meta_gene.loc[top_genes, "Markers"]
    inst_ser = inst_ser[inst_ser != "N.A."]
    ser_counts = inst_ser.value_counts()
    max_count = ser_counts.max()
    max_cat = "_".join(sorted(ser_counts[ser_counts == max_count].index.tolist()))
    max_cat = max_cat.replace(" marker", "").replace(" ", "-")
    print(inst_cluster, max_cat)
    meta_leiden.loc[inst_cluster, "Cell_Type"] = max_cat

# rename clusters
meta_leiden["name"] = meta_leiden.apply(lambda x: x["Cell_Type"] + "_" + x["leiden"], axis=1)
leiden_names = meta_leiden["name"].values.tolist()
meta_leiden.index = leiden_names

# transfer cell type labels to single cells
leiden_to_cell_type = deepcopy(meta_leiden)
leiden_to_cell_type.set_index("leiden", inplace=True)
leiden_to_cell_type.index.name = None

adata_combined.obs["Cell_Type"] = adata_combined.obs["leiden"].apply(lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "Cell_Type"])
adata_combined.obs["Cluster"] = adata_combined.obs["leiden"].apply(lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "name"])


adatas = {}
for batch in batches:
    subgroup_mask = adata_combined.obs['batch'] == batch  # Create a boolean mask for the current group
    subgroup_data = adata_combined[subgroup_mask].copy()  # Create a copy of the data for the subgroup
    adatas[batch] = subgroup_data  # Store the subgroup data in a dictionary


# %% Spatial scatter plot for specific gene expressions
sq.pl.spatial_scatter(adatas[batches[1]], library_id="spatial", shape=None, cmap="Reds",
                      color=["CEMIP2", "FLT1", "NOTCH1", "NRP1", "PECAM1", "RNF144B", "STAT3", "THSD4"])
plt.tight_layout()
plt.show()


# %% Spatial scatter plot for Cell-Types
sc.pl.umap(adatas[batch_id], color=["Cell_Type"], legend_fontsize=10, legend_loc='on data')
sq.pl.spatial_scatter(adatas[batch_id], shape=None, color="Cell_Type", library_id="spatial", figsize=(10, 10))
plt.show()

sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color= ['Cluster'])
plt.show()


# %%
adata = adatas[batch_id]
# adata.uns['spatial']=adata.obsm["spatial"]
all_tumor_clusters = [x for x in meta_leiden.index.tolist() if "Tumor" in x]
sig_leiden.columns = meta_leiden.index.tolist()
ser_egfr = sig_leiden[all_tumor_clusters].loc["EGFR"]
egfr_high = ser_egfr[ser_egfr > 0].index.tolist()
egfr_high_cells = [adata.obs.iloc[x]['cell_id'] for x in range(len(adata.obs)) if adata.obs.iloc[x]['Cluster'] in egfr_high]
egfr_low = ser_egfr[ser_egfr <= 0].index.tolist()
sq.pl.spatial_scatter(adatas[batch_id], groups=egfr_high, library_id="spatial", shape=None, color= ['Cluster'])
plt.show()



# %% Building the spatial neighbors graphs
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
# Compute centrality scores
sq.gr.centrality_scores(adata, cluster_key="leiden")
sq.pl.centrality_scores(adata, cluster_key="leiden", figsize=(16, 5))
plt.show()

# Compute custom centrality scores - adata.obs is used and the scores are stored in adata.uns
sq.gr.centrality_scores(adata, cluster_key="Cell_Type")
sq.pl.centrality_scores(adata, cluster_key="Cell_Type", figsize=(16, 5)) # Load scores
plt.show()


# %% Compute co-occurrence probability
adata_subsample = sc.pp.subsample(adata, fraction=0.5, copy=True)
sq.gr.co_occurrence(adata_subsample, cluster_key="leiden")
sq.pl.co_occurrence(adata_subsample, cluster_key="leiden", clusters="0", figsize=(10, 10))
sq.pl.spatial_scatter(adata_subsample, library_id="spatial", shape=None, color= ['Cluster'])
plt.show()

_, idx = adata.obsp["spatial_connectivities"].nonzero()
sq.pl.spatial_scatter(
    adata[idx, :],
    library_id="spatial",
    color="leiden",
    connectivity_key="spatial_connectivities",
    size=3,
    edges_width=1,
    edges_color="black",
    img=False,
    title="K-nearest neighbors",
    shape=None)
plt.show()

# %% Neighbors enrichment analysis
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata_subsample, cluster_key="leiden", figsize=(8, 8), title="Neighborhood enrichment adata")
plt.show()


# %% Compute Moran's I score
sq.gr.spatial_neighbors(adata_subsample, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(adata_subsample, mode="moran", n_perms=100, n_jobs=1)
target_genes = adata_subsample.uns["moranI"].head(10).index

sq.pl.spatial_scatter(adata, color=[target_genes[0]], shape=None, cmap="Reds")
plt.show()