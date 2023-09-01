 # %% Import modules
import numpy as np
import pandas as pd
import commot as ct

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq


# %% Load data

adata = sc.read_10x_h5(filename="/home/ext_choi_yoonho_mayo_edu/jupyter/cell_feature_matrix.h5")
df = pd.read_csv("/home/ext_choi_yoonho_mayo_edu/jupyter/cells.csv.gz")

df.set_index(adata.obs_names, inplace=True)
adata.obs = df.copy()
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
adata.obs

sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)

cprobes = (adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100)
cwords = (adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100)
print(f"Negative DNA probe count % : {cprobes}")
print(f"Negative decoding count % : {cwords}")

# %% Plot distribution of total transcripts per cell
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].set_title("Total transcripts per cell")
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[1].set_title("Unique transcripts per cell")
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axs[1])
axs[2].set_title("Area of segmented cells")
sns.histplot(adata.obs["cell_area"], kde=False, ax=axs[2])
axs[3].set_title("Nucleus ratio")
sns.histplot(adata.obs["nucleus_area"] / adata.obs["cell_area"], kde=False, ax=axs[3])
plt.show()

# %% Filter cell and genes
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=5)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %% Visulaize annotation on UMAP and spatial coordinates
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "leiden"])

sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["leiden"], wspace=0.4)
plt.show()



# %% Check common_marker_genes and meta genes
df_ref_panel_ini = pd.read_excel('/home/ext_choi_yoonho_mayo_edu/jupyter/Xenium_HumanBrainPanel_GeneList_removed_double_columns.xlsx', index_col=0)
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
ser_counts = adata.obs["leiden"].value_counts()
ser_counts.name = "cell counts"
meta_leiden = pd.DataFrame(ser_counts)
cat_name = "leiden"
sig_leiden = pd.DataFrame(columns=adata.var_names, index=adata.obs[cat_name].cat.categories)
for clust in adata.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = adata[adata.obs[cat_name].isin([clust]), :].X.mean(0)
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

adata.obs["Cell_Type"] = adata.obs["leiden"].apply(lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "Cell_Type"])
adata.obs["Cluster"] = adata.obs["leiden"].apply(lambda x: leiden_to_cell_type.loc["Leiden-" + str(x), "name"])


sq.pl.spatial_scatter(adata, library_id="spatial", color=["CEMIP2", "FLT1", "NOTCH1", "NRP1", "PECAM1", "RNF144B",
                                                          "STAT3", "THSD4"], shape=None, cmap="Reds")
plt.tight_layout()
plt.show()


# adata.uns['spatial']=adata.obsm["spatial"]
all_tumor_clusters = [x for x in meta_leiden.index.tolist() if "Tumor" in x]
sig_leiden.columns = meta_leiden.index.tolist()
ser_egfr = sig_leiden[all_tumor_clusters].loc["EGFR"]
egfr_high = ser_egfr[ser_egfr > 0].index.tolist()
egfr_high_cells = [adata.obs.iloc[x]['cell_id'] for x in range(len(adata.obs)) if adata.obs.iloc[x]['Cluster'] in egfr_high]
egfr_low = ser_egfr[ser_egfr <= 0].index.tolist()

sc.pl.umap(adata, color=["Cell_Type"], legend_fontsize=10, legend_loc='on data')
sq.pl.spatial_scatter(adata, shape=None, color="Cell_Type", library_id="spatial", figsize=(10, 10))
plt.show()

sq.pl.spatial_scatter(adata, groups=egfr_high_cells, color="Cell_Type", cmap="Reds")
#TODO: 특정 cluster만 spatial scatter하는 방법 알아 볼 것
sq.pl.spatial_scatter(adata, groups=egfr_high, color="Cluster")
plt.show()

# %% Building the spatial neighbors graphs
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
# Compute centrality scores
sq.gr.centrality_scores(adata, cluster_key="leiden")
sq.pl.centrality_scores(adata, cluster_key="leiden", figsize=(16, 5))
plt.show()

# Compute custom centrality scores - adata.obs is used and the scores are stored in adata.uns
sq.gr.centrality_scores(adata, cluster_key="Cell_Type_leiden")
sq.pl.centrality_scores(adata, cluster_key="Cell_Type_leiden", figsize=(16, 5)) # Load scores
plt.show()


# %% Compute co-occurrence probability
adata_subsample = sc.pp.subsample(adata, fraction=0.5, copy=True)
sq.gr.co_occurrence(adata_subsample, cluster_key="leiden")
sq.pl.co_occurrence(adata_subsample, cluster_key="leiden", clusters="12", figsize=(10, 10))
sq.pl.spatial_scatter(adata_subsample, color="leiden")
plt.show()


# %% Neighbors enrichment analysis
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata, cluster_key="leiden", figsize=(8, 8), title="Neighborhood enrichment adata")
plt.tight_layout()
plt.show()

# %% Compute Moran's I score
sq.gr.spatial_neighbors(adata_subsample, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(adata_subsample, mode="moran", n_perms=100, n_jobs=1)
target_genes = adata_subsample.uns["moranI"].head(10).index

sq.pl.spatial_scatter(adata, color=["VEGFA"], shape=None, cmap="Reds")
plt.tight_layout()
plt.show()