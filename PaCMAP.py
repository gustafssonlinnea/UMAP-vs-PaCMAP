import pacmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit


class PaCMAP:
    def __init__(self, n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0):
        """
        n_neighbors=None leads to default choice:
        - For dataset whose sample size < 10000: 10
        - For dataset whose sample size n > 10000: 10 + 15 * (log10(n) - 4)
        """
        self.reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)  
        
    def fit(self, X):
        # X_transformed = embedding.fit_transform(X, init='pca')
        self.embedding = self.reducer.fit_transform(X)
    
    def get_embedding(self):
        return self.embedding
    
    def plot_embedding(self, y, data_title):
        assert hasattr(self, 'embedding'), "PaCMAP not fitted to data. Call the 'fit' method first."
        
        sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
        num_classes = len(np.unique(y))
        
        if self.embedding.shape[1] == 2:  # Create 2D scatter plot
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=y, cmap='Spectral', s=5)
            
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5).set_ticks(np.arange(num_classes))
            plt.title(f'PaCMAP projection of the {data_title} dataset', fontsize=24)
            plt.show()
        
        elif self.embedding.shape[1] == 3:  # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot

            # Scatter plot
            scatter = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], self.embedding[:, 2], c=y, cmap='Spectral', s=5)
            ax.set_aspect('equal', 'datalim') # Set aspect ratio

            # Add colorbar
            colorbar = fig.colorbar(scatter, ax=ax, boundaries=np.arange(num_classes + 1) - 0.5)
            colorbar.set_ticks(np.arange(num_classes))

            # Set title
            ax.set_title(f'PaCMAP projection of the {data_title} dataset', fontsize=24)
            plt.show()


"""# =============== Step 4 ===============
# visualize the embedding
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)"""