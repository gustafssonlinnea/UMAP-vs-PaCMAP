import pacmap
from CONSTANTS import *


class PaCMAP_reducer:
    def __init__(self, n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0):
        """
        n_neighbors=None leads to default choice:
        - For dataset whose sample size < 10000: 10
        - For dataset whose sample size n > 10000: 10 + 15 * (log10(n) - 4)
        """
        self.reducer = pacmap.PaCMAP(n_components=n_components, 
                                     n_neighbors=n_neighbors, 
                                     MN_ratio=MN_ratio, 
                                     FP_ratio=FP_ratio,
                                     random_state=RANDOM_SEED)  
        
    def fit(self, X):
        self.embedding = self.reducer.fit_transform(X)
    
    def get_embedding(self):
        return self.embedding
