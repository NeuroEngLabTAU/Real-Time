import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import threading
import multiprocessing as mp

from XtrRT.data import Data #data collection
from XtrRT.viz import Viz  #real time raw data plotting
# from XtrRT.ica import Viz_ICA   # interactive ica in real time
from XtrRT.ica_streaming import Viz_ICA_Streaming    # Streaming ICA
from XtrRT.electrodes_raw_streaming import Electrodes_Raw_Streaming  # stream raw data with electrodes location

#for image load, electrode selection, and heatmaps
import cv2
import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path

import os
import time


#########################
###### Load image #######
#########################
def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width


#create a dummy heatmap
def fill_polygon(vertices, num_rows, num_cols):
    # Create a grid of points
    x = np.linspace(0, num_cols-1, num_cols)
    y = np.linspace(0, num_rows-1, num_rows)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.flatten(), yy.flatten())).T

    # Create a path from the polygon vertices
    path = Path(vertices)

    # Determine whether each point is inside the polygon
    mask = path.contains_points(points)
    mask = mask.reshape(num_rows, num_cols)

    # Fill the grid with random values between 0 and 1, and NaN values outside the polygon
    result = np.empty((num_rows, num_cols))
    result[~mask] = np.nan
    result[mask] = np.random.rand(np.sum(mask))

    return result


def run_data_collection(data):
    data.start()
    while True:
        time.sleep(1)  # Sleep for 1 second to avoid excessive CPU usage
        # Perform any other data processing or analysis here
        # You can update the shared data object for the figures to access

def viz_raw(raw_flag, data, filters):
    if raw_flag:
        raw_fig, raw_axes = plt.figure()
        raw_streaming = Viz(data=data, window_secs=10, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False,
                            filters=filters,
                            update_interval_ms=10, ylim_exg=(-250, 250), max_points=None, max_timeout=15,
                            filter_data=True, figure=raw_fig, axes=raw_axes)
        raw_streaming.start()

def viz_electrodes(electrodes_flag, data, filters, x_coor, y_coor, width, height, image):
    if electrodes_flag:
        electrodes_fig, electrodes_axes = plt.subplots()
        elctrodes_streaming = Electrodes_Raw_Streaming(data=data, window_secs=5, plot_exg=True, plot_imu=False,
                                                       plot_ica=False,
                                                       find_emg=False, filters=filters, update_interval_ms=100,
                                                       ylim_exg=(-250, 250), max_points=None, max_timeout=15,
                                                       x_coor=x_coor, y_coor=y_coor, width=width, height=height,
                                                       image=image, filter_data=True,
                                                       figure=electrodes_fig, axes=electrodes_axes)

        elctrodes_streaming.start()

if __name__ == '__main__':

    # define the desired visualisation
    raw = False  # raw signal streaming
    # viz_ica = False
    # viz_ica_streaming = False  # streaming ICA signals with heatmaps
    electrodes = True  # raw signal electrodes

    # prepare constants:
    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf")
    data.start()

    # upload your image - insert image path

    # image_path = r"C:\Users\YH006_new\Simulation\Paul_45.jpg"

    # Get the path of the current script
    script_path = os.path.abspath(__file__)
    # Get the directory containing the script
    script_directory = os.path.dirname(script_path)
    # Construct the image path relative to the script directory
    image_path = os.path.join(script_directory, "face-muscles-anatomy.jpg")

    image, height, width = image_load(image_path)

    # to select the electrodes locations on the image leave empty, otherwise load
    # x_coor=[]
    # y_coor=[]
    x_coor = [557, 398, 336, 444, 466, 342, 389, 490, 601, 450, 335, 328, 422, 545, 551, 689]
    y_coor = [836, 786, 690, 721, 657, 634, 586, 599, 567, 541, 537, 477, 387, 381, 290, 289]
    # x_coor = np.load(r'C:\Users\YH006_new\Simulation\x_coor_45.npy')
    # y_coor = np.load(r'C:\Users\YH006_new\Simulation\y_coor_45.npy')
    # if len(x_coor)==0:  # if manually inserted coordinates, there is not need to get the electrode locations (otherwise select electrodes from image)
    #     get_location()

    # create dummy heatmap for funcAnimator setup
    vertices = list(zip(x_coor, y_coor))
    num_rows, num_cols = height, width
    d_interpolate = fill_polygon(vertices, num_rows, num_cols)

    data.add_annotation("Start recording")

    filters = {'highpass': {'W': 30}, 'comb': {'W': 50}}


    # Create separate processes for each figure
    raw_process = mp.Process(target=viz_raw, args=(raw, data, filters,))
    electrodes_process = mp.Process(target=viz_electrodes, args=(electrodes, data, filters, x_coor, y_coor, width, height, image,))

    # Start the processes
    raw_process.start()
    electrodes_process.start()

    # Wait for the processes to finish
    raw_process.join()
    electrodes_process.join()

    # plt.show()

    data.add_annotation("Stop recording")
    data.stop()

    print(data.annotations)
    print('process terminated')
