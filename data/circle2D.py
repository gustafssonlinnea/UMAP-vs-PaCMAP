import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './utils')
from CONSTANTS import *
import utils
import seaborn as sns


class Circle2D:
    def __init__(self, circle_factor=30, n_points=100):
        self.theta = np.linspace(0, 2 * np.pi, n_points)
        self.x = circle_factor * np.cos(self.theta)
        self.y = circle_factor * np.sin(self.theta)
        c = np.linspace(0, 1, n_points)
        bin_edges = np.linspace(0, 1, n_points // 5)
        self.c = np.digitize(c, bin_edges, right=True)

    def plot_circle(self, fig_path='figures', fig_name='fig_Circle2D'):
        plt.figure()
        plt.scatter(self.x, self.y, c=self.c, cmap=CYCLIC_CMAP, linewidth=0)
        plt.title('Circle2D')
        
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        
        plt.axis('equal')
        plt.gca().set_facecolor(BG_COLOR)
        
        # Remove lines around plot
        sns.despine(left=True, bottom=True)
        
        path = f'{fig_path}/Circle2D'
        utils.create_directory_if_not_exists(path)
        plt.savefig(f'{path}/{fig_name}.png', dpi=300)
        print(f'\nSuccessfully saved Circle2D figure')

    def load_data(self):
        X = np.column_stack((self.x, self.y))  # (n_points, 2)
        y = self.c  # (n_points, )
        return X, y

if __name__ == '__main__':
    circle = Circle2D()
    circle.plot_circle()
    X, y = circle.load_data()
