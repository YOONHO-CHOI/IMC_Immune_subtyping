# %% Import modules and functions
import os, glob, time, natsort, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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

def find_column(x):
    return '+'.join(x.index[x == 1].tolist())

def get_windows(job, n_neighbors):
    '''
    For each region and each individual cell in dataset, return the indices of the nearest neighbors.
    'job:  metadata containing the start time,index of region, region name, indices of region in original dataframe
    n_neighbors:  the number of neighbors to find for each cell
    '''
    start_time, idx, tissue_name, indices = job
    job_start = time.time()
    print("Starting: {}/{} :{}".format((idx + 1), len(exps), (exps[idx])))
    tissue = tissue_group.get_group(tissue_name)
    to_fit = tissue.loc[indices][[x, y]].values

    #     fit = NearestNeighbors(n_neighbors=n_neighbors+1).fit(tissue[[X,Y]].values)
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[x, y]].values)
    m = fit.kneighbors(to_fit)
    #     m = m[0][:,1:], m[1][:,1:]
    m = m[0], m[1]

    # sort_neighbors
    args = m[0].argsort(axis=1)
    add = np.arange(m[1].shape[0]) * m[1].shape[1]
    sorted_indices = m[1].flatten()[args + add[:, None]]
    neighbors = tissue.index.values[sorted_indices]
    end_time = time.time()

    print("Finishing: {}/{} : {},{},{}".format((idx + 1),len(exps),exps[idx], end_time - job_start,
          end_time - start_time))
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
coord_col =["centroid-0","centroid-1"]
markers   =['155Gd-FoxP3_dichotomized','170Er-CD3_dichotomized', '148-Pan-Ker_dichotomized']
root      = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/'

# Load preprocessed data (This includes immune subtypes)
immune_sub_df = pd.read_csv(root+'hyp_immune_sub.csv').iloc[:,1:]
immune_sub_df = immune_sub_df.dropna(subset=['Immune_ph_x']) # Get rid of NaN in 'H&E+IHC based immune subtypes'

# Load expr data and get patient list
expr_dir  = root+'visualization/data/dich_expr'
expr_list = glob.glob(os.path.join(expr_dir, '*.csv'))
expr_pats = [pat.split('/')[-1] for pat in expr_list]

# Load coord data and get patient list
coord_dir= root+'visualization/data/regionprops/'
coord_list= glob.glob(os.path.join(coord_dir, '*.csv'))
coord_pats= [pat.split('/')[-1] for pat in coord_list]

# Get common patient list from expr & coord patient lists (this list has exact filenames including region)
common_pats  = np.unique([x for x in expr_pats if x in coord_pats])

# Extract immune subtype data for common patient list
rows=[]
for idx, row in immune_sub_df.iterrows(): # For all rows in immune_sub_df, we will extract common-pats records
    if 'LN' in row['File name']: # We do not consider lymph nodes
        continue
    else:
        temp_pat = row['File name'].split('.')[0] # Get patient name without extension
        print(temp_pat)
        for pat in common_pats: # For all common patient list
            if temp_pat in pat: # check the row(immune_sub_df) is included in common patient list
                temp_row = row  # if the row is included, we keep the row (temp_row)
                temp_row['File name'] = pat # Replace filename with the
                temp_row['region'] = pat.split('_ac_')[0][-1]
                rows.append(temp_row.reset_index(drop=True))
df = pd.concat(rows, axis=1).T.reset_index(drop=True)
df.columns = list(immune_sub_df.columns)+['region']
grouped_df = df.groupby('medical_record_number')

# %% Load expr and coord files to add cell locations
DF = []
marker_x =['155Gd-FoxP3_dichotomized','170Er-CD3_dichotomized', '148-Pan-Ker_dichotomized']
case       = 1
all_region = 1
for group_name, group_data in grouped_df:
    print(group_name)
    for idx, row in group_data.iterrows():
        fname    = row['File name']
        expr_df  = pd.read_csv(os.path.join(expr_dir,fname))
        coord_df = pd.read_csv(os.path.join(coord_dir,fname))
        print(fname)
        temp_df  = pd.DataFrame()
        temp_df[['centroid-0','centroid-1']] = coord_df[['centroid-0','centroid-1']]
        temp_df[markers] = expr_df[markers]
        temp_df.rename(columns={'170Er-CD3_dichotomized': 'CD3',
                                '141-SMA_dichotomized': 'SMA',
                                '143Nd-Vimentin_dichotomized': 'Vimentin',
                                '148-Pan-Ker_dichotomized': "PanCK",
                                '155Gd-FoxP3_dichotomized': 'FoxP3'}, inplace=True)

        temp_df, dict = make_table_and_palette(temp_df, marker_x, combination=True)
        name_series = temp_df['celltype']
        celltype = union_markers(name_series, ['Vimentin', 'SMA'], 'Stroma')
        remove = ['CD3+PanCK', 'FoxP3+CD3+PanCK']
        for target in remove:
            celltype = celltype.replace(target, 'None')
        temp_df.celltype = celltype
        temp_df = temp_df[temp_df['celltype'] != 'None']

        # add columns
        temp_df['File name'] = fname
        temp_df['Immune_ptype'] = row['Immune_ph_x']
        temp_df['region'] = row['region']
        temp_df['all_region'] = all_region
        temp_df['case'] = case
        DF.append(temp_df)
        all_region = all_region+1
    case = case+1
DF = pd.concat(DF)
DF = DF[['File name','case','region','all_region','celltype','Immune_ptype','centroid-0','centroid-1']]
DF = DF.reset_index(drop=True)
# %% CN feature extraction
"""
-ks = individual window sizes to collect.  
      -This accelerates the process of trying different window sizes by collecting them all in one step.
-reg:  column to use for region (should be different for every tissue)
-file_type('csv','pickle'):  file type
-cluster_col : column name of cell type cluster to be used for neighborhoods
-keep_cols:  columns to transfer from original dataframe to dataframe with windows and neighborhoods

원래는 cell에 해당되는 모든 pixel들로부터 다수를 보이는 cell type을 매칭해야 함.
그러나 내가 가진 데이터는 1개의 cell(object)에 매칭되는 여러 marker들이 있어 하나의 cell type을 고려키 어려움.
데이터의 차이가 있는 상황에서 neighborhood identification을 하는것엔 여러 옵션이 가능.
1. 대표적인 marker에 국한하여 cell typing.
   이 경우 다음의 코드를 고려중.
   marker_x=['148-Pan-Ker','155Gd-FoxP3','170Er-CD3']
   Z = pd.merge(df_expr[['Object']+marker_x], df_coord)
2. marker들의 조합까지 type으로 고려하여 cell typing.
우선 2번 먼저 해보고자 함.
"""
ks = [10] # k=5 means it collects 5 nearest neighbors for each center cell
reg = 'all_region'
x = 'centroid-0'
y = 'centroid-1'
cluster_col = 'celltype'
keep_cols = [x,y,reg,cluster_col]
save_path = ''

# Read in data and do some quick data rearrangement
n_neighbors = max(ks)
cells = pd.concat([DF,pd.get_dummies(DF[cluster_col])],1)

#cells = cells.reset_index() #Uncomment this line if you do any subsetting of dataframe such as removing dirt etc or will throw error at end of next next code block (cell 6)
sum_cols = cells[cluster_col].unique()
values = cells[sum_cols].values

#find windows for each cell in each tissue region
tissue_group = cells[[x,y,reg]].groupby(reg)
exps = list(cells[reg].unique())
tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)]
tissues = [get_windows(job,n_neighbors) for job in tissue_chunks]

"""
# for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
out_dict = {}
for k in ks:
    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

# concatenate the summed windows and combine into one dataframe for each window size tested.
windows = {}
for k in ks:
    window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0], index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols],window],1)
    windows[k] = window
"""
out_dict = {}
windows = {}
for k in ks:
    windows_df = []
    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        """
        neighbors: Area에 속한 cell만을 대상, 그 각각에 대한 nearst neighbor hood를 sort 후 기록한 결과 array (#cells X #n) 
        values: 전체 area를 대상으로 모든 cell에 대하여 cell type이 onehot encodding된 matrix ((#cells X # area) X #ctype)
        window: Area에 속한 cell들 각각에 대하여 k개의 neighborhood를 구하고 그들의 cell type에 대하여 sum을 수행함 (#cells X #ctype)
                "Area내 각 cell들을 대상으로  neighborhood들의 cell type을 summation"
        """
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)
        df = pd.DataFrame(out_dict[(tissue_name, k)][0], index=out_dict[(tissue_name, k)][1].astype(int), columns=sum_cols)
        windows_df.append(df)
    window = pd.concat(windows_df)
    window_idx = window.index
    cells_idx  = cells.loc[window_idx]
    window = pd.concat([cells_idx[keep_cols], window])
    windows[k] = window.reset_index(drop=True)



#%% MBKM
k = 10
n_neighborhoods = 10
neighborhood_name = "neighborhood"+str(k)
k_centroids = {}

windows2 = windows[10]
# windows2[cluster_col] = cells[cluster_col]

km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)

labelskm = km.fit_predict(windows2[sum_cols].values)
k_centroids[k] = km.cluster_centers_
cells['neighborhood10'] = labelskm
cells[neighborhood_name] = cells[neighborhood_name].astype('category')
#
# cell_order = ['tumor cells', 'CD11c+ DCs', 'tumor cells / immune cells',
#        'smooth muscle', 'lymphatics', 'adipocytes', 'undefined',
#        'CD4+ T cells CD45RO+', 'CD8+ T cells', 'CD68+CD163+ macrophages',
#        'plasma cells', 'Tregs', 'immune cells / vasculature', 'stroma',
#        'CD68+ macrophages GzmB+', 'vasculature', 'nerves',
#        'CD11b+CD68+ macrophages', 'granulocytes', 'CD68+ macrophages',
#        'NK cells', 'CD11b+ monocytes', 'immune cells',
#        'CD4+ T cells GATA3+', 'CD163+ macrophages', 'CD3+ T cells',
#        'CD4+ T cells', 'B cells']

# this plot shows the types of cells (ClusterIDs) in the different niches (0-7)
k_to_plot = 10
niche_clusters = (k_centroids[k_to_plot])
tissue_avgs = values.mean(axis = 0)
fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
fc = pd.DataFrame(fc,columns = sum_cols)
sns.clustermap(fc.loc[[0,1,2,3,4,5,6,7,8,9],fc.columns], vmin =-3,vmax = 3,cmap = 'bwr',row_cluster = False)
plt.show()
# s.savefig("raw_figs/celltypes_perniche_10.pdf")

cells['neighborhood10'] = cells['neighborhood10'].astype('category')
sns.lmplot(data = cells,x = 'centroid-0',y='centroid-1',hue = 'neighborhood10',palette = 'bright',height = 8,col = reg,col_wrap = 10,fit_reg = False)
plt.show()

