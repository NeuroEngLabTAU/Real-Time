import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


#for image load, electrode selection, and heatmaps
import cv2
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Circle  # for marking the electrodes on the image

import os
def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width
class FaceAndElectrodesPlot:

    def __init__(self, image, x_coor, y_coor, num_channels=16, window_secs=10, fs=100):
        self.image = image
        self.x_coor = x_coor
        self.y_coor = y_coor
        self.num_channels = num_channels
        self.window_secs = window_secs
        self.fs = fs

        self.xdata = np.linspace(0, self.window_secs, self.window_secs * self.fs)
        self.ydata = np.array([np.sin(2 * np.pi * (i + 1) * self.xdata) for i in range(self.num_channels)]).T

        self.figure, self.axs = self.setup_figure()

    def setup_figure(self):
        rows, cols = 4, 8
        fig, axs = plt.subplots(rows, cols, figsize=(15, 8))

        for i in range(self.num_channels):
            row, col = divmod(i, cols // 2)
            # Plot sine wave
            ax_wave = axs[row, col * 2]
            ax_wave.plot(self.xdata, self.ydata[:, i], color='tab:blue', lw=0.5)
            ax_wave.set_title(f'Channel {i+1}')
            ax_wave.set_ylim([-1, 1])

            # Plot image with circle
            ax_img = axs[row, col * 2 + 1]
            ax_img.imshow(self.image, aspect='equal')  # Preserve aspect ratio
            circle = Circle((self.x_coor[i], self.y_coor[i]), 15, edgecolor='red', linewidth=2, fill=False)
            ax_img.add_patch(circle)
            ax_img.axis('off')

        plt.tight_layout()
        return fig, axs

    def debug_with_grid(self):
        # Function to overlay a grid for debugging
        fig, ax = plt.subplots()
        ax.imshow(self.image, aspect='equal')
        ax.grid(True)
        ax.set_xticks(np.arange(0, self.image.shape[1], 10))
        ax.set_yticks(np.arange(0, self.image.shape[0], 10))
        plt.show()

# Example usage
num_channels = 16
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
X_COOR = np.random.uniform(100, 200, num_channels)  # Example x coordinates
Y_COOR = np.random.uniform(100, 200, num_channels)  # Example y coordinates
image_path = os.path.join(script_directory, "face-muscles-anatomy.jpg")
image, height, width = image_load(image_path)

plotter = FaceAndElectrodesPlot(image, X_COOR, Y_COOR, num_channels)
plotter.debug_with_grid()  # Call this function to see the grid for alignment
plt.show()
