import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './utils')
from CONSTANTS import *
import utils

class Spiral3D:
    def __init__(self, spiral_factor=30, n_points=400):
        self.z = np.linspace(0, 1, n_points)
        self.x = self.z * np.sin(spiral_factor * self.z)
        self.y = self.z * np.cos(spiral_factor * self.z)
        c = np.linspace(0, 1, n_points)
        bin_edges = np.linspace(0, 1, n_points // 5)  #  // 10
        self.c = np.digitize(c, bin_edges, right=True)
        
        
    def plot_spiral(self, fig_path='figures', fig_name='fig_Spiral3D'):
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.scatter(self.x, self.y, self.z, c=self.c, 
                   cmap=CONTINUOUS_CMAP, linewidth=0)

        # Change the color of the panes
        ax.xaxis.set_pane_color(BG_COLOR)
        ax.yaxis.set_pane_color(BG_COLOR)
        ax.zaxis.set_pane_color(BG_COLOR)

        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_title('Spiral3D')
        path = f'{fig_path}/Spiral3D'
        utils.create_directory_if_not_exists(path)
        plt.savefig(f'{path}/{fig_name}.png', dpi=300)
        print(f'\nSuccessfully saved Spiral3D figure')
    
    def load_data(self):
        X = np.column_stack((self.z, self.x, self.y))  # (n_points, 3)
        y = self.c  # (n_points, )
        return X, y
    
if __name__ == '__main__':
    spiral = Spiral3D()
    spiral.plot_spiral()
    X, y = spiral.load_data()
