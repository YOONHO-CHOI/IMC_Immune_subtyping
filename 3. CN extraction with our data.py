# %% Import modules and functions
import os, glob, time, natsort, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def plot_hyperion(df, marker_x, unique_lst, save_dir=None, fname=None):
    """
    The primary colors are correspond to every single elements in marker_x.
    Based on the primary colors, this code automatically set colors to combinations of markers.
    """
    elements = df.columns[:len(marker_x)]  # Same as numbers of elements (PanCK, FoxP3, CD3)
    elements = natsort.natsorted(elements)
    primary_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    dict = {}
    # for None
    if len(np.where(unique_lst == 'None')[0]):
        none = np.where(unique_lst == 'None')[0][0]
        dict[unique_lst[none]] = (0, 0, 0)
    # for primary colors
    if len(elements) <= len(primary_colors):
        for i, element in enumerate(elements):
            dict[element] = primary_colors[i]

    # for combination
    for i, names in enumerate(unique_lst):
        name_list = names.split('+')
        if len(name_list) == len(elements):
            dict[names] = (1, 1, 1)
        elif len(name_list) != 1:
            combination = 0
            for name in name_list:
                combination = combination + np.array(dict[name])
            combination = combination / combination.max()
            combination = tuple(combination)
            dict[names] = combination

    plt.style.use('dark_background')  # default, dark_background
    sns.scatterplot(y="centroid-0", x="centroid-1", hue='new', linewidth=0, data=df, s=5, palette=dict)
    plt.style.use('dark_background')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    plt.axis("off")
    plt.tight_layout()

    if save_dir != None:
        plt.savefig(os.path.join(save_dir, fname))
        plt.clf()
    else:
        plt.show()
        plt.clf()

def make_table_and_palette(df, marker_x, combination=False):
    """
    The primary colors are correspond to every single elements in marker_x.
    Based on the primary colors, this code automatically set colors to combinations of markers.
    """
    elements = df.columns[2:2+len(marker_x)]  # Same as numbers of elements (PanCK, FoxP3, CD3)
    df_elements = df[elements]
    primary_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    names = []
    for i, row in df_elements.iterrows():
        name_elements = (elements * row.values).values
        name_elements = [x for x in name_elements if x]
        name = '+'.join(name_elements)
        if combination == False and len(name_elements)!=1:
            name = ''
        names.append(name)

    df['celltype'] = names
    df = df.replace(r'^\s*$', "None", regex=True)
    dict = {}

    # for None
    lst_un = natsort.natsorted(df['celltype'].unique())
    if len(np.where(np.asarray(lst_un) ==  "None")[0]):
        none = np.where(np.asarray(lst_un) == 'None')[0][0]
        dict[lst_un[none]] = (0, 0, 0)

    # for primary colors
    if len(elements) <= len(primary_colors):
        for i, element in enumerate(elements):
            dict[element] = primary_colors[i]

    if combination:
        # for combination
        for i, names in enumerate(lst_un):
            name_list = names.split('+')
            if len(name_list) == len(elements):
                dict[names] = (1, 1, 1)
            elif len(name_list) != 1:
                combination = 0
                for name in name_list:
                    combination = combination + np.array(dict[name])
                combination = combination / combination.max()
                combination = tuple(combination)
                dict[names] = combination
    return df, dict

def get_windows(job, max_neighbors, reg_list):
    '''
    For each region and each individual cell in dataset, return the indices of the nearest neighbors.
    job:  metadata containing the start time, index of region, region name, indices of region in original dataframe
    n_neighbors:  the number of neighbors to find for each cell
    '''
    start_time, idx, tissue_name, indices = job
    job_start = time.time()
    print("Starting: {}/{} - region: {}".format((idx + 1), len(reg_list), (reg_list[idx])))
    tissue = tissue_group.get_group(tissue_name)
    to_fit = tissue.loc[indices][['centroid-0','centroid-1']].values

    neigh = NearestNeighbors(n_neighbors=max_neighbors).fit(tissue[['centroid-0','centroid-1']].values)
    m = neigh.kneighbors(to_fit)

    # sort_neighbors
    args = m[0].argsort(axis=1)
    add = np.arange(m[1].shape[0]) * m[1].shape[1]
    sorted_indices = m[1].flatten()[args + add[:, None]]
    neighbors = tissue.index.values[sorted_indices]
    end_time = time.time()

    print("Run time:{}, Total run time:{}".format(end_time - job_start, end_time - start_time))
    return neighbors.astype(np.int32)

def union_markers(name_series, marker_name, union_name):
    """ Change the marker names and their combinations to union-name in a series """
    combinations = []
    for i in range(1, len(marker_name) + 1):
        for subset in itertools.permutations(marker_name, i):
            combinations.append(list(subset))
    for marker_list in combinations:
        target_marker_name = '+'.join(marker_list)
        name_series = name_series.replace(target_marker_name, union_name)
    return name_series

# %% Settings
root = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/'
DF   = pd.read_csv(os.path.join(root, 'data', '1.processed', 'Unified_DF_for_CN.csv'))

# %% CN feature extraction
"""
-ks : Individual window sizes to collect. This uses different window sizes by collecting them all in one step.
-reg:  column to use for region (should be different for every tissue)
-cluster_col : column name of cell type cluster to be used for neighborhoods
-keep_cols:  columns to transfer from original dataframe to dataframe with windows and neighborhoods

원래는 cell에 해당되는 모든 pixel들로부터 다수를 보이는 cell type을 매칭해야 함.
그러나 내가 가진 데이터는 1개의 cell(object)에 매칭되는 여러 marker들이 있어 하나의 cell type을 고려키 어려움.
데이터의 차이가 있는 상황에서 neighborhood identification을 하는것엔 여러 옵션이 가능.
"""
# Settings
K  = [10] # Nearest neighbors for each center cell
max_neighbors = max(K)
reg_col       = 'all_region'
cluster_col   = 'celltype'
keep_cols     = ['centroid-0','centroid-1','Immune_ptype',reg_col,cluster_col]
cells         = pd.concat([DF,pd.get_dummies(DF[cluster_col])],1) # DF + multi hot vectors according to cell types
celltype_cols = cells[cluster_col].unique()
values        = cells[celltype_cols].values

# Find windows for each cell in each tissue region
tissue_group  = cells[['centroid-0','centroid-1',reg_col]].groupby(reg_col)
reg_list      = list(cells[reg_col].unique())
tissue_chunks = [(time.time(),reg_list.index(t),t,a)
                 for t,indices in tissue_group.groups.items()
                 for a in np.array_split(indices,1)] # 1 chunk = (time, region_list idx, actual region, cells idx)
tissues       = [get_windows(job, max_neighbors, reg_list) for job in tissue_chunks]

# for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
# concatenate the summed windows and combine into one dataframe for each window size tested.
out_dict = {}
windows = {}
for k in K:
    windows_df = []
    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        """
        neighbors: Area에 속한 cell만을 대상, 그 각각에 대한 nearst neighbor hood를 sort 후 기록한 결과 array (#cells X #n) 
        values: 전체 area를 대상으로 모든 cell에 대하여 cell type이 onehot encodding된 matrix ((#cells X # area) X #ctype)
        window: Area에 속한 cell들 각각에 대하여 k개의 neighborhood를 구하고 그들의 cell type에 대하여 sum을 수행함 (#cells X #ctype)
                Area내 각 cell들을 대상으로  neighborhood들의 cell type을 summation.
                neighborhood 중 어떤 cell type이 얼만큼 있는지, 각 cell 별로 구함. (#cells X # ctype) <- element value= count 
        df: celltype_cols에 대하여 앞서구한 window 값들로 채워진 dataframe.
        windows_df: k neighborhoods를 대상으로 구해지는 df들의 모음.
        window: cells의 정보와 windows_df를 결합한 dataframe.
        """
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(celltype_cols)).sum(axis=1)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)
        df = pd.DataFrame(out_dict[(tissue_name, k)][0], index=out_dict[(tissue_name, k)][1].astype(int), columns=celltype_cols)
        windows_df.append(df)
    window = pd.concat(windows_df)
    window_idx = window.index
    cells_idx  = cells.loc[window_idx]
    window = pd.concat([cells_idx[keep_cols], window], axis=1)
    windows[k] = window.reset_index(drop=True)


#%% MBKM
k = 10
neighborhood_name = "neighborhood"+str(k)
temp_windows = windows[k]

# Get CN features
num_CN = 10 # Set number of CN features you want.
k_centroids     = {}
km = MiniBatchKMeans(n_clusters = num_CN,random_state=0)
labelskm = km.fit_predict(temp_windows[celltype_cols].values)
k_centroids[k] = km.cluster_centers_
cells['neighborhood10'] = labelskm
cells[neighborhood_name] = cells[neighborhood_name].astype('category')

niche_clusters = (k_centroids[num_CN])
tissue_avgs = values.mean(axis = 0)
# 전체 tissue로부터 각 Cell type에 대해 조사한 mean Neighborhood 값과 비교하여, Enrichment score를 계산
fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
fc = pd.DataFrame(fc,columns = celltype_cols)



# %% Cluster heatmap plot
sns.clustermap(fc.loc[np.arange(num_CN),fc.columns], vmin =-3,vmax = 3,cmap = 'bwr',row_cluster = False)
plt.show()

# %% Entire scatter plot
cells['neighborhood10'] = cells['neighborhood10'].astype('category')
sns.lmplot(data = cells,x = 'centroid-0',y='centroid-1',hue = 'neighborhood10',
           palette = 'bright',height = 8,col = reg_col,col_wrap = 10,fit_reg = False)
plt.show()

# %% Scatter plots
cell_groups = cells.groupby('all_region')
z = cells[cells.all_region==1]
plt.style.use('dark_background')  # default, dark_background
sns.scatterplot(y="centroid-0", x="centroid-1", hue='neighborhood10', linewidth=0, data=z, s=5, palette='bright')  # , legend=False)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.axis("off")
plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(save_dir, fname))
plt.clf()

# %% Boxplot
# %% Boxplot
cell_groups  = cells.groupby(['Immune_ptype'])
for subtype, group in tissue_group.groups.items():
    print(subtype)

import scipy.stats as stats
p_values = []
labels = cells['Immune_ptype'].unique()
# Perform statistical comparisons and store the p-values
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        subset1 = cells[cells['Immune_ptype']==labels[i]]
        subset2 = cells[cells['Immune_ptype'] == labels[j]]
        _, p_value = stats.ttest_ind(subset1[celltype_cols], subset2[celltype_cols])
        p_values.append(p_value)

# Plot the boxplots
cells.boxplot(column=list(celltype_cols), by='Immune_ptype')

# Add a title
plt.title('Boxplot Comparison')

# Add a y-axis label
plt.ylabel('Values')

# Add p-value annotations
annotation_y = cells[celltype_cols].values.max()  # Set the y-coordinate for annotations
for i, p_value in enumerate(p_values):
    x = i+1  # Set the x-coordinate for annotations
    plt.annotate(f'p-value = {p_value:.3f}', xy=(x, annotation_y), ha='center')

# Show the plot
plt.show()
