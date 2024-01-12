import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
from flexitext import flexitext
from matplotlib.patches import FancyArrowPatch

#fig, ax = plt.subplots(figsize=(10, 10))
#ax.scatter(age_at_mission, year_of_mission)

def plot_embedding(embedding, y, method_name, data_title):
    plt.rcParams.update({"font.family": "Corbel", "font.weight": "light"})
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.labelcolor"] = "white"
    plt.rcParams["ytick.labelcolor"] = "white"     
    sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    num_classes = len(np.unique(y))

    if embedding.shape[1] == 2:  # Create 2D scatter plot
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
        
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5).set_ticks(np.arange(num_classes))
        plt.title(f'UMAP projection of the {data_title} dataset', fontsize=24)
        plt.show()
    
    elif embedding.shape[0] == 3:  # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot

        # Scatter plot
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=y, cmap='Spectral', s=5)
        ax.set_aspect('equal', 'datalim') # Set aspect ratio

        # Add colorbar
        colorbar = fig.colorbar(scatter, ax=ax, boundaries=np.arange(num_classes + 1) - 0.5)
        colorbar.set_ticks(np.arange(num_classes))

        # Set title
        ax.set_title(f'{method_name} projection of the {data_title} dataset', fontsize=24)
        plt.show()
    
def plot_embeddings(y, data_title, dot_size=100, alpha=0.5, fig_path='figures', fig_name='figure', *embeddings):            
    #sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    num_classes = len(np.unique(y))
    
    num_plots = len(embeddings)
    size = 10
    fig, axs = plt.subplots(1, num_plots, figsize=(size * num_plots, size), squeeze=False)
    fig.suptitle(f'Dimensionality reduction of the {data_title} dataset', fontsize=28)
    
    #for method_name, embedding in embeddings.items():
    for i in range(num_plots):
        ax = axs[0][i]
        method_name, embedding = next(iter(embeddings[i].items()))

        if embedding.shape[1] == 2:  # Create 2D scatter plot
            
            ax.scatter(embedding[:, 0], 
                       embedding[:, 1], 
                       c=y, 
                       alpha=alpha, 
                       cmap='Spectral', 
                       s=dot_size, 
                       linewidth=0)
            
            #ax.colorbar(boundaries=np.arange(num_classes + 1) - 0.5).set_ticks(np.arange(num_classes))
            # ax.set(adjustable='box', aspect='equal')
            # ax.set_aspect('equal', 'box')
            ax.set_box_aspect(1)
            
            # Set title
            ax.set_title(f'{method_name}', fontsize=24)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        elif embedding.shape[0] == 3:  # Create a 3D scatter plot
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot

            # Scatter plot
            scatter = ax.scatter(embedding[:, 0], 
                                 embedding[:, 1], 
                                 embedding[:, 2], 
                                 c=y, 
                                 alpha=alpha, 
                                 cmap='Spectral', 
                                 s=dot_size, 
                                 linewidth=0)
            

            # Add colorbar
            #colorbar = fig.colorbar(scatter, ax=ax, boundaries=np.arange(num_classes + 1) - 0.5)
            #colorbar.set_ticks(np.arange(num_classes))
            
            # ax.set(adjustable='box', aspect='equal')
            ax.set_box_aspect(1)

            # Set title
            ax.set_title(f'{method_name}', fontsize=24)
    
    fig.tight_layout(pad=3)
    # plt.subplots_adjust(wspace=0, hspace=0)
    
    #plt.savefig(f'{fig_path}/{fig_name}.png', bbox_inches='tight', pad_inches=0.5)
    plt.savefig(f'{fig_path}/{fig_name}.png')
    print(f'Successfully saved figure')
    # plt.show()