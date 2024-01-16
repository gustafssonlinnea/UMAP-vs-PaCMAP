from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit


class UMAP_reducer:
    def __init__(self, min_dist=0.1, n_components=2, n_epochs=500, n_neighbors=1000):
        self.reducer = UMAP(min_dist=min_dist, n_components=n_components, n_epochs=n_epochs, n_neighbors=n_neighbors)       
        
    def fit(self, X):
        self.reducer.fit(X)
    
    def get_embedding(self):
        return self.reducer.embedding_
    
    def plot_embedding(self, y, data_title):
        assert hasattr(self.reducer, 'embedding_'), "UMAP reducer not fitted to data. Call the 'fit' method first."
        
        sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
        
        embedding = self.reducer.embedding_
        num_classes = len(np.unique(y))

        if self.reducer.n_components == 2:  # Create 2D scatter plot
            plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
            
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5).set_ticks(np.arange(num_classes))
            plt.title(f'UMAP projection of the {data_title} dataset', fontsize=24)
            plt.show()
        
        elif self.reducer.n_components == 3:  # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot

            # Scatter plot
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=y, cmap='Spectral', s=5)
            ax.set_aspect('equal', 'datalim') # Set aspect ratio

            # Add colorbar
            colorbar = fig.colorbar(scatter, ax=ax, boundaries=np.arange(num_classes + 1) - 0.5)
            colorbar.set_ticks(np.arange(num_classes))

            # Set title
            ax.set_title(f'UMAP projection of the {data_title} dataset', fontsize=24)
            plt.show()
