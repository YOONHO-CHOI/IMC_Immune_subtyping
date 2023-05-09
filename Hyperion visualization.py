#%% Import modules
import os, fnmatch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#%%
def plot_hyperion(df, marker_x, save_dir=None, fname=None):
    """
    The primary colors are correspond to every single elements in marker_x.
    Based on the primary colors, this code automatically set colors to combinations of markers.
    """
    import natsort
    elements = df.columns[:len(marker_x)]  # Same as numbers of elements (PanCK, FoxP3, CD3)
    elements = natsort.natsorted(elements)
    primary_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    dict = {}
    # for None
    if len(np.where(lst_un == 'None')[0]):
        none = np.where(lst_un == 'None')[0][0]
        dict[lst_un[none]] = (0, 0, 0)
    # for primary colors
    if len(elements) <= len(primary_colors):
        for i, element in enumerate(elements):
            dict[element] = primary_colors[i]

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

    plt.style.use('dark_background')  # default, dark_background
    sns.scatterplot(y="centroid-0", x="centroid-1", hue='new', linewidth=0, data=df, s=5,
                    palette=dict)  # , legend=False)
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

def find_column(x):
    return '+'.join(x.index[x == 1].tolist())

#%%
coord_column=["centroid-0","centroid-1"]

marker_x=['170Er-CD3_dichotomized', '156Gd-CD4_dichotomized','161Dy-CD20_dichotomized',
          '155Gd-FoxP3_dichotomized','162Dy-CD8a_dichotomized', '148-Pan-Ker_dichotomized']

p_ids=['REQ31365_OTL7_Tumor','REQ30820_20210503_0NVB','REQ30820_20210503_0NVD',"REQ30820_20210503_0NVF","REQ31365_OTL9_TUMOR"]
sub_type=["IN_foxp3_high","IN_foxp3_low","IN_foxp3_low","IE","ID"]
marker_x=['155Gd-FoxP3_dichotomized','170Er-CD3_dichotomized', '148-Pan-Ker_dichotomized']
palette ={'170Er-CD3_dichotomized': to_rgba("r",0.3), '148-Pan-Ker_dichotomized': to_rgba("b",0.3),'155Gd-FoxP3_dichotomized': to_rgba("yellow",0.3)}
out_dir="spatial_images/"

n=0
save_dir = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/results'
os.makedirs(save_dir, exist_ok=True)

DF = []
for p_id,sub in zip(p_ids,sub_type):
    # plot_ROI_sp(p_id,marker_x,out_dir,palette,sub)
    path = "/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/dich_expr"
    path_coord = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/regionprops/'
    path_mask = '/home/ext_choi_yoonho_mayo_edu/jupyter/Project/IMC/visualization/data/cpout/masks/'

    ROI_p = fnmatch.filter(os.listdir(path), p_id + "*")
    l = len(ROI_p)
    coord_p = fnmatch.filter(os.listdir(path_coord), p_id + "*")
    # fig, axes = plt.subplots(l, 1, sharex=True, figsize=(20, l + 10))

    for i, ROI in enumerate(ROI_p):
        print(ROI)
        df_expr = pd.read_csv(path + "/" + ROI)
        df_coord = pd.read_csv(path_coord + "/" + ROI)
        img_mask_path = path_mask + ROI.split(".csv")[0] + ".tiff"
        print(img_mask_path)
        df = pd.DataFrame()
        df[marker_x] = df_expr[marker_x]
        df.rename(columns={'155Gd-FoxP3_dichotomized': 'FoxP3', '170Er-CD3_dichotomized': 'CD3',
                           '148-Pan-Ker_dichotomized': "PanCK"}, inplace=True)
        df['new'] = df.apply(find_column, axis=1)
        df = df.replace(r'^\s*$', "None", regex=True)
        df[coord_column] = df_coord[coord_column]
        lst_un = df['new'].unique()
        lst_un = np.sort(lst_un)
        plot_hyperion(df, marker_x)
        plot_hyperion(df, marker_x, save_dir, str(n)+'_new.png')
        n = n + 1