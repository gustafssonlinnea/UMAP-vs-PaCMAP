from umap.umap_ import UMAP
from CONSTANTS import *


class UMAP_reducer:
    def __init__(self, min_dist=0.1, n_components=2, n_epochs=500, n_neighbors=1000):
        self.reducer = UMAP(min_dist=min_dist, 
                            n_components=n_components, 
                            n_epochs=n_epochs, 
                            n_neighbors=n_neighbors,
                            random_state=RANDOM_SEED)       
        
    def fit(self, X):
        self.reducer.fit(X)
    
    def get_embedding(self):
        return self.reducer.embedding_