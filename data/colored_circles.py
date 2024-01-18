import sys
sys.path.insert(0, 'utils')

import matplotlib.pyplot as plt
import numpy as np
import utils


class ColoredCircles:
    def __init__(self, points_per_class=50, image_size=100, data_path='data/colored_circles'):
        self.num_classes = 4
        self.points_per_class = points_per_class
        self.image_size = image_size
        self.dpi = plt.rcParams['figure.dpi']  # pixel in inches

        # Create a list to store the image data
        self.image_data = []

        self.data_path = data_path
        utils.create_directory_if_not_exists(data_path)
        
        c = 0.3
        self.colors = [(0.0, 0.0, 0.0), (c, 0.0, 0.0), (c * 2, 0.0, 0.0), (c * 3, 0.0, 0.0)]
        
        self.create_dataset()
        print('Dataset created')
    
    def create_dataset(self):
        # Generate and save a separate plot for each data point
        for class_label in range(1, self.num_classes + 1):
            for i in range(self.points_per_class):
                # Create a new figure and axis for each data point
                fig, ax = plt.subplots(figsize=(self.image_size / self.dpi, self.image_size / self.dpi))
                #ax.set_xlim([0, self.image_size])
                #ax.set_ylim([0, self.image_size])

                # Random radius for each circle
                radius = np.random.uniform(1, self.image_size / 2)

                # Plot the circle
                circle = plt.Circle((self.image_size / 2, 
                                    self.image_size / 2), 
                                    radius, 
                                    edgecolor='none', 
                                    facecolor=self.colors[class_label - 1])
                ax.add_patch(circle)

                # Set aspect ratio and remove axis labels
                ax.set_aspect('equal', 'box')
                ax.axis('off')

                # Save the figure with a unique name (e.g., based on class label and index)
                file_name = f'figure_class_{class_label}_point_{i}.png'
                images_path = f'{self.data_path}/images'
                utils.create_directory_if_not_exists(images_path)
                image_path = f'{images_path}/{file_name}'
                fig.savefig(image_path, bbox_inches='tight', pad_inches=0)

                # Convert the image to a NumPy array
                img_array = plt.imread(image_path)

                # Append the image array to the list
                self.image_data.append(img_array)

                plt.close()

        # Convert the list of image arrays to a NumPy array
        image_data_array = np.array(self.image_data)

        # Save the NumPy array
        np.save(f'{self.data_path}/image_data.npy', image_data_array)

    def get_dataset(self):
        path = f'{self.data_path}/image_data.npy'
        # Load the NumPy array containing image data
        image_data_array = np.load(path)

        # Generate labels (y)
        y = np.repeat(np.arange(1, self.num_classes + 1), self.points_per_class)
        
        # Get the total number of data points
        num_data_points = self.num_classes * self.points_per_class# image_data_array.shape[0]

        # Reshape the image data array to match the expected format (num_samples, height, width, channels)
        X = image_data_array.reshape(num_data_points, image_data_array.shape[1], image_data_array.shape[2], 4)

        # Display the shapes of X and y
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        return X, y

cc = ColoredCircles()
X, y = cc.get_dataset()