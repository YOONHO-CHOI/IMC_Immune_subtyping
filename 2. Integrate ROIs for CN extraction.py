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
expr_dir  = root+'visualization/data/dich_expr_gaus95'
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
DF.to_csv(os.path.join(root, 'data', '1.processed', 'Unified_DF_for_CN.csv'), index=False)