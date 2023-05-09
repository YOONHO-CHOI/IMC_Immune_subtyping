#%% Import modules
import os, fnmatch, natsort, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Functions
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

    df['new'] = names
    df = df.replace(r'^\s*$', "None", regex=True)
    dict = {}

    # for None
    lst_un = natsort.natsorted(df['new'].unique())
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

# def plot_hyperion(df, marker_x, save_dir=None, fname=None, combination=False, sort = False):
def plot_hyperion(df, dict, save_dir=None, fname=None):
    plt.style.use('dark_background')  # default, dark_background
    sns.scatterplot(y="centroid-0", x="centroid-1", hue='new', linewidth=0, data=df, s=5, palette=dict)  # , legend=False)
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

#%%
coord_column=["centroid-0","centroid-1"]
p_ids   = ['REQ31365_OTL7_Tumor','REQ31365_OTL9_TUMOR','REQ30820_20210503_0NVB','REQ30820_20210503_0NVD',
           'REQ30820_20210503_0NVF','31365_0UHD-tumor','31365_0UHF-tumor','31365_0UHh-tumor','31365_0UHJ-tumor']
sub_type= ["IN_foxp3_high","ID","IN_foxp3_low","IN_foxp3_low","IE","ID","IN_foxp3_low","ID","ID"]

path       = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/dich_expr'
path_mask  = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/cpout/masks/'
path_coord = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/regionprops/'


out_dir="spatial_images/"
save_dir = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/remove_CD3PanCK_Foxp3CD3PanCK'
os.makedirs(save_dir, exist_ok=True)
marker_x =['155Gd-FoxP3_dichotomized','170Er-CD3_dichotomized', '148-Pan-Ker_dichotomized']
DF = []
n  =1
for p_id,sub in zip(p_ids,sub_type):
    ROI_p = fnmatch.filter(os.listdir(path), p_id + "*")
    coord_p = fnmatch.filter(os.listdir(path_coord), p_id + "*")

    for i, ROI in enumerate(ROI_p):
        print(ROI)
        df_expr = pd.read_csv(path + "/" + ROI)
        df_coord = pd.read_csv(path_coord + "/" + ROI)
        img_mask_path = path_mask + ROI.split(".csv")[0] + ".tiff"

        df = pd.DataFrame()
        df[coord_column] = df_coord[coord_column]
        df[marker_x] = df_expr[marker_x]
        df.rename(columns={'170Er-CD3_dichotomized': 'CD3', '143Nd-Vimentin_dichotomized': 'Vimentin',
                           '148-Pan-Ker_dichotomized': "PanCK", '141-SMA_dichotomized': 'SMA',
                           '155Gd-FoxP3_dichotomized': 'FoxP3'}, inplace=True) #'141-SMA_dichotomized': 'SMA',
        df, dict = make_table_and_palette(df, marker_x, combination=True)

        name_series = df.new
        new = union_markers(name_series, ['Vimentin', 'SMA'], 'Stroma')
        remove = ['CD3+PanCK', 'FoxP3+CD3+PanCK']
        dict = {key: value for key, value in dict.items() if key not in remove}
        # dict = {'CD3':(1,0,0), 'Stroma':(0,1,0), 'PanCK':(0,0,1), 'CD3+FoxP3':(1,1,1), 'None':(0,0,0)}
        for target in remove:
            new = new.replace(target, 'None')
        df.new = new
        # plot_hyperion(df, dict)
        temp_save_dir = os.path.join(save_dir, sub)
        os.makedirs(temp_save_dir, exist_ok=True)
        name = '_'.join(['pat'+str(n)] + ROI.split('.')[0].split('_')[3:6])
        plot_hyperion(df, dict, temp_save_dir, name+'.png')
        df_filtered = df[df['new'] != 'None']
        column_stats = df_filtered.groupby('new')['new'].describe(include='all').T.reset_index(drop=True).iloc[0]

    n = n + 1