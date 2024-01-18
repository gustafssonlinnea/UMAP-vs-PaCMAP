import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys
sys.path.insert(0, './utils')
from CONSTANTS import *

plt.rc('text', usetex=True)

    
def plot_embeddings(embeddings,
                    y,
                    n_neighbors,
                    data_title,
                    dot_size=100, 
                    alpha=0.5, 
                    fig_path='figures', 
                    fig_name='figure', 
                    cmap=DISCRETE_CMAP,
                    metric='KNN'):
    """Creates and saves a 2D scatter plot of embeddings"""
    num_classes = len(np.unique(y))
    num_plots = len(embeddings)
    size = 10
    fig, axs = plt.subplots(1, num_plots, figsize=(size * num_plots, size), squeeze=False)
    fig.suptitle(f'Dimensionality reduction of the {data_title} dataset: \
        Embeddings with the best {metric} accuracy', fontsize=28)
    
    if cmap not in list(mpl.colormaps):
        cmap = ListedColormap(sns.color_palette(cmap))

    # Remove lines around plot
    sns.despine(left=True, bottom=True)

    for i in range(num_plots):
        ax = axs[0][i]
        method_name, embedding = next(iter(embeddings[i].items()))
                    
        sns.scatterplot(x=embedding[:, 0], 
                        y=embedding[:, 1], 
                        hue=y, 
                        palette=cmap, 
                        alpha=alpha, 
                        s=dot_size,
                        linewidth=0, 
                        ax=ax,
                        legend=False)

        ax.set_box_aspect(1)
        ax.set_title(f'{method_name}', fontsize=24)
        ax.set_facecolor(BG_COLOR)
        
        # Turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_xlabel(r'$n=' + str(n_neighbors[i]) + '$', fontsize=20)
    
    fig.tight_layout(pad=3)
    plt.savefig(f'{fig_path}/{fig_name}.png')
    print(f'\nSuccessfully saved {metric} accuracy figure ' \
        f'of {data_title} dataset')
