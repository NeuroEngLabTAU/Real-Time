import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import threading

from XtrRT.data import Data  #data collection
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
import datetime
import time
import keyboard
import sys


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


#######################################
###### select electrode location ######
#######################################

def click_event(event, x, y, flags, params):
    global x_coor
    global y_coor
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        x_coor.append(x)
        y_coor.append(y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '(' + str(x) + ',' +
                    str(y) + ')', (x, y), font,
                    0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

    return x_coor, y_coor


def get_location():
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


# create a dummy heatmap
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


# function for video recording
def record_video(webcam):
    global is_recording_video, output_filename

    # Get the current time including milliseconds
    recording_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")[:-3]
    output_filename = f"video_{recording_start_time}.avi"
    print(f"Video recording started at {recording_start_time} ...")

    # Define the video writer object
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

    # Start recording until user interrupts or is_recording is False
    while is_recording_video:
        ret, frame = webcam.read()  # Read frame from webcam

        if ret:
            # Write the frame to the output video file
            out.write(frame)
        else:
            print("Failed to capture frame")
            break

    print("Video recording stopped at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    # Release the video writer object
    out.release()


# functions to control video recording
def start_video(data, cap):
    global is_recording_video
    if not is_recording_video:
        is_recording_video = True
        record_thread = threading.Thread(target=record_video, args=(cap,))
        record_thread.start()
        data.add_annotation('started video recording')


def stop_video(data):
    global is_recording_video
    if is_recording_video:
        is_recording_video = False
        data.add_annotation('stopped video recording')


if __name__ == '__main__':

    # define the desired visualisation
    viz_raw = False  # Aaron's real time raw data plotting
    # viz_ica = False  # Bara's semi real-time ICA
    viz_ica_streaming = False  # streaming ICA signals with heatmaps
    Electrodes_raw = True  # raw signal electrodes

    ica_integration_time = 10  # seconds
    stop_ica = False  # if True the ica will be calculated for all the recorded data (ica_integration_time will be ignored),
    # until the button 'ICA converged!' is pressed, then the data will be plotted according to the unmixing matrix
    # that was calculated last (when the button 'ICA converged!' was pressed)
    window_secs = 10  # seconds

    # if usage of video recording is desired:
    # takes about 20 seconds to connect to the webcam at the beginning
    video = True  # if True video recording is controlled by the space bar for start and 'q' for stop
    one_video = True  # starts the first video recording with the EMG recording,
    # also can set only one_video to True if only one video of the whole EMG recording is desired

    if video or one_video:
        print("connecting to webcam...")
        cap = cv2.VideoCapture(0)
        print("connected to webcam at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        # Check if the webcam is opened successfully
        if not cap.isOpened():
            print("Failed to open webcam")
            cap.release()  # Release the webcam resource
            sys.exit()

        # Define the video writer object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_filename = ""
        is_recording_video = False


    if viz_ica_streaming or Electrodes_raw:
        #upload your image - insert image path
        image_path = r"C:\Users\YH006_new\Simulation\Paul_45.jpg"
        # image_path = 'face-muscles-anatomy.jpg'

        image, height, width = image_load(image_path)


        #to select the electrodes locations on the image leave empty, otherwise load
        # x_coor=[]
        # y_coor=[]
        # x_coor = [557, 398, 336, 444, 466, 342, 389, 490, 601, 450, 335, 328, 422, 545, 551, 689]
        # y_coor = [836, 786, 690, 721, 657, 634, 586, 599, 567, 541, 537, 477, 387, 381, 290, 289]
        x_coor = np.load(r'C:\Users\YH006_new\Simulation\x_coor_45.npy')
        y_coor = np.load(r'C:\Users\YH006_new\Simulation\y_coor_45.npy')
        if len(x_coor)==0:  # if manually inserted coordinates, there is not need to get the electrode locations (otherwise select electrodes from image)
            get_location()

        # create dummy heatmap for funcAnimator setup
        vertices = list(zip(x_coor, y_coor))
        num_rows, num_cols = height, width
        d_interpolate = fill_polygon(vertices, num_rows, num_cols)

    # prepare the data object
    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf")
    data.start()

    while not data.has_data:  # Wait to start collecting data before doing anything
        continue

    start_time = datetime.datetime.now()
    print('started data recording at: ', start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    data.add_annotation("Start recording")

    if video:
        # press 'space' to start the video recording and 'q' to stop it
        keyboard.on_press_key("space", lambda _: start_video(data, cap))
        keyboard.on_press_key("q", lambda _: stop_video(data))

    if one_video:
        start_video(data, cap)


    filters = {'highpass': {'W': 30}, 'comb': {'W': 50}}

    if viz_raw:
        original_viz = Viz(data, window_secs=window_secs, plot_exg=True, plot_imu=True, plot_ica=False, find_emg=False, filters=filters,
                  update_interval_ms=10, ylim_exg=(-250, 250), ylim_acc=(-1.1, 1.1), max_points=None, max_timeout=15)

        original_viz.start()

    # if viz_ica:
    #     viz = Viz_ICA(data, window_secs=10, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
    #               update_interval_ms=10, ylim_exg=(-250, 250), max_points=None, max_timeout=15,
    #               x_coor=x_coor, y_coor=y_coor, width=width, height=height, image=image, filter_data=True)

    if viz_ica_streaming:
        ica_viz = Viz_ICA_Streaming(data, window_secs=window_secs, ica_integration_time=ica_integration_time, stop_ica=stop_ica,
                                    plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
                                    update_interval_ms=200, ylim_exg=(-5, 5), max_points=None, max_timeout=15,
                                    x_coor=x_coor, y_coor=y_coor, width=width, height=height, image=image, d_interpolate=d_interpolate, filter_data=True)

        ica_viz.start()



    if Electrodes_raw:
        elctrodes_streaming = Electrodes_Raw_Streaming(data, window_secs=window_secs, plot_exg=True, plot_imu=False, plot_ica=False,
                   find_emg=False, filters=filters, update_interval_ms=10, ylim_exg=(-250, 250), max_points=None, max_timeout=15,
                   x_coor=x_coor, y_coor=y_coor, width=width, height=height, image=image, filter_data=True)

        elctrodes_streaming.start()

    if viz_raw or viz_ica_streaming or Electrodes_raw:
        plt.show()

    time.sleep(1)
    # to stop the recording press esc
    print('press \'esc\' to stop data recording')

    while True:
        if keyboard.is_pressed('esc'):
            break
        time.sleep(0.1)

    if (video or one_video) and is_recording_video:  # if recording is still on, stop the video
        stop_video(data)

    total_time = datetime.datetime.now() - start_time
    data.add_annotation("Stop recording at " + f"{int(total_time.total_seconds() // 60):02d}:"
                                               f"{int(total_time.total_seconds() % 60):02d}."
                                               f"{total_time.microseconds // 1000:03d}")
    data.stop()
    print('stopped data recording at: ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    print(data.annotations)
    print('process terminated')
