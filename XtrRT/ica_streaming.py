import warnings

import sklearn.exceptions
from sklearn.decomposition import FastICA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import ImageGrab
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from .data import Data, ConnectionTimeoutError
from random import randrange
from scipy.signal import butter, lfilter, filtfilt, iirnotch #for filtering the data
from matplotlib.widgets import Button, TextBox #for button in funcanimator

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time

#for heatmap
from numpy.linalg import inv
from scipy.interpolate import griddata
import cv2
import matplotlib.image as mpimg

#for ICA
from picard import picard #*need to cite*,  install "python-picard"

class Viz_ICA_Streaming:

    def __init__(self,
                 data: Data,
                 window_secs: float = 10,
                 plot_exg: bool = True,
                 plot_imu: bool = True,
                 plot_ica: bool = True,
                 ylim_exg: tuple = (-100, 100),
                 ylim_imu: tuple = (-1, 1),
                 update_interval_ms: int = 200,
                 max_points: (int, None) = 1000,
                 max_timeout: (int, None) = 15,
                 find_emg: bool = False,
                 filters: dict = None,
                 x_coor: np.ndarray[np.float64] = None,
                 y_coor: np.ndarray[np.float64] =None,
                 width: int = None,
                 height: int=None,
                 image: np.ndarray[np.uint8] = None,
                 d_interpolate: np.ndarray[np.float64] =None,
                 filter_data: bool=False):

        assert plot_exg or plot_imu

        self.data = data
        self.plot_exg = plot_exg
        self.plot_imu = plot_imu
        self.plot_ica = plot_ica if plot_ica and plot_exg else False
        self.window_secs = window_secs
        self.axes = None
        self.figure = None
        self.ylim_exg = ylim_exg
        self.ylim_imu = ylim_imu
        self.xdata = None
        self.ydata = None
        self.lines = []
        self.pause_time= None
        self.unpause_time = None
        self.bg = None
        self.last_exg_sample = 0
        self.last_imu_sample = 0
        self.init_time = datetime.now()
        self.update_interval_ms = update_interval_ms
        self.max_points = max_points
        self.new_max_points=max_points
        self._backend = None

        self.pause=False

        self.fs=None #added sampling rate as attribute  (fs)

        # for the buttons on the figure:
        self.ica_integration_time = 10  # default value of 10 seconds for ICA integration time
        self.button_start = None
        self.start_label = 'Start'
        self.start_time = None
        self.timer = None
        self.timer_ax = None
        self.timer_text = None
        self.timer_running = False


        #define the arrangement order of  the atlas
        self.wanted_order = np.array([12, 13, 14, 15,
                                      11, 10, 9, 8,
                                      4, 5, 6, 7,
                                      3, 2, 1, 0])

        #obtain the image characteristics
        self.y_coor = y_coor
        self.x_coor = x_coor
        self.width = width
        self.height = height
        self.grid_x = None
        self.grid_y = None
        self.points = None
        self.image = image

        self.d_interpolate=d_interpolate #dummy heatmap for funcanimator initilization

        #added the option to filter the data
        self.filter_data=filter_data


        # Confirm initial data retrieval before continuing (or raise Error if timeout)
        while not (self.data.is_connected and self.data.has_data):
            plt.pause(0.01)
            if (datetime.now() - self.init_time).seconds > max_timeout:
                if not data.is_connected:
                    raise ConnectionTimeoutError
                elif not data.has_data:
                    raise TimeoutError(f"Did not succeed to stream data within {max_timeout} seconds.")

        self.setup()


    def setup(self):

        self.ylim_imu = self.ylim_imu if self.ylim_imu else self.ylim_imu
        self.ylim_exg = self.ylim_exg if self.ylim_exg else self.ylim_exg

        # set the sampling_rate
        self.fs = self.data.fs_exg if self.plot_exg else self.data.fs_imu
        # set the grid_x, grid_y, and points
        self.grid_y, self.grid_x = np.mgrid[1:self.height + 1, 1:self.width + 1]
        self.points = np.column_stack((self.x_coor, self.y_coor))

        # Get data
        n_exg_samples, n_exg_channels = self.data.exg_data.shape if self.plot_exg else (0, 0)  # TODO: can be None if imu data comes first
        n_imu_samples, n_imu_channels = self.data.imu_data.shape if self.plot_imu else (0, 0)

        # Make timestamp vector
        max_samples = max((n_imu_samples, n_exg_samples))
        last_sec = max_samples / self.fs
        ts_max = self.window_secs if max_samples <= self.window_secs*self.fs else last_sec
        ts_min = ts_max - self.window_secs
        ts = np.arange(ts_min, ts_max, 1/self.fs)

        #
        n_channels = n_exg_channels + n_imu_channels
        self.xdata = ts
        self.ydata = np.full((len(ts), n_channels), np.nan)
        self.max_points = self.max_points if self.window_secs*self.fs < len(self.xdata) else self.window_secs*self.fs

        # For auto-maximization of figure
        #screensize = ImageGrab.grab().size
        #px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        #screensize_inches = [px*npx for npx in screensize]

        # set up the grid for the heatmaps with sources
        row_num, col_num = 4, 12 #define the row and col
        ratio = round(self.height / self.width, 1)

        #define layout for the plot
        fig, axes = plt.subplots()
        spec = fig.add_gridspec(row_num, col_num)

        axs = []
        for i in range(row_num):
            for j in range(col_num):
                if (j == 2 or j == 5 or j == 8 or j == 11):
                    ax = fig.add_subplot(spec[i, j])
                    axs.append(ax)
                elif (j == 0 or j == 3 or j == 6 or j == 9):
                    ax = fig.add_subplot(spec[i, j:j + 2])
                    axs.append(ax)
        axes.axis('off')


        source = 0
        for j in range(len(axs)):
            source = int(j / 2)
            if (j % 2 != 0):
                axs[j].imshow(self.image)
                im=axs[j].pcolormesh(self.d_interpolate, cmap='jet', alpha=0.5)
                self.lines.append(im)
                axs[j].set_aspect('auto')
                axs[j].axis('off')
            else:
                line, =axs[j].plot(self.xdata, self.ydata[:, source], color='tab:blue', lw=0.5)
                axs[j].margins(0)
                axs[j].set_ylim(self.ylim_exg)
                self.lines.append(line)
                axs[j].set_title('source %d' % (self.wanted_order[source] + 1), fontsize=15)
            axs[j].xaxis.set_ticklabels([])
            axs[j].xaxis.set_ticks([])

            # only show the y-axis for the left-most column
            if (j % 8 == 0):
                axs[j].tick_params(axis='y', labelsize=10, direction='in', length=4, width=1, bottom=False, labelbottom=False)
            else:
                axs[j].tick_params(axis='y', left=False, labelleft=False, bottom=False, labelbottom=False)




        self.axes = axs
        self.figure = fig

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()


        fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.95, hspace=0.5)

        self.figure.canvas.mpl_connect('close_event', self.close)


        self.bg = [self.figure.canvas.copy_from_bbox(ax.bbox) for ax in np.ravel(self.axes)] if self._backend != 'module://mplopengl.backend_qtgl' else None


    @staticmethod
    def _correct_matrix(matrix, desired_samples):

        if matrix.shape[0] < desired_samples:
            n_to_pad = desired_samples - matrix.shape[0]
            matrix = np.pad(matrix.astype(float), ((0, n_to_pad), (0, 0)), mode='constant', constant_values=np.nan)

        return matrix

    @staticmethod
    def _crop(matrix, desired_samples):

        nsamples = matrix.shape[0]
        n_samples_cropped = nsamples - desired_samples
        data = matrix[n_samples_cropped:]

        return data, n_samples_cropped


    #functions used for atlas
    #find the array + size of array cooresponding to the red/orange (depending on the threshold) of the heatmsp
    @staticmethod
    def _red_size(arr, thresh):
        myrange = np.nanmax(arr) - np.nanmin(arr)
        norm_arr = (arr - np.nanmin(arr)) / myrange
        # threshold for red/orange 0.75-0.1
        condition = (norm_arr.ravel()) > thresh
        flat_indices = np.where(condition)[0]

        return flat_indices, flat_indices.shape[0]

    #normalized an array from 0(min) to 1(max)
    @staticmethod
    def _norm(arr):
        myrange = np.nanmax(arr) - np.nanmin(arr)
        norm_arr = (arr - np.nanmin(arr)) / myrange
        return norm_arr


    #given a ditionary containing the smear/cluster charactersitics
    # the function orders the area of the smear, and well as providing the keys corresponding to that order
    @staticmethod
    def _combine_and_sort(dictionary):
        combined_list = []
        key_mapping = []

        for key in dictionary.keys():
            size = dictionary[key].size
            combined_list.append(size)
            key_mapping.append(key)

        sorted_indices = sorted(range(len(combined_list)), key=lambda k: combined_list[k], reverse=False)
        combined_list_sorted = [combined_list[i] for i in sorted_indices]
        key_mapping_sorted = [key_mapping[i] for i in sorted_indices]

        return combined_list_sorted, key_mapping_sorted

    # Smear/cluster class, contains the characteristics of the cluster/smear
    class Smear:
        def __init__(self, number):
            self.number = number
            self.not_noise = 0

    #find the atlas/ordering of the sources according to a pre-defined order (wanted_order)
    def atlas(self, W, K, number_of_electrode, wanted_order):
        # plot the number_of_electrode in a way that makes sense
        inverse = np.absolute(inv(np.matmul(W, K)))

        order_electrode = np.full((number_of_electrode,), -1)
        f_interpolate = []
        for i in range(number_of_electrode):
            # f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))
            interpolate_data = griddata(self.points, inverse[:, i], (self.grid_x, self.grid_y), method='linear')

            norm_arr = Viz_ICA_Streaming._norm(interpolate_data)
            f_interpolate.append(norm_arr)

            # find atlas
            data = f_interpolate[i]
            # find red point
            electrode = np.argmax(inverse[:, i])
            order_electrode[i] = electrode

        #########################
        ###     Atlas     #######
        #########################
        # wondering if it will be faster to just check red for all 16.....
        # check if duplicates (usually means one is noise/noiser)
        unique, counts = np.unique(order_electrode, return_counts=True)
        duplicates = unique[counts > 1]
        # the potential electrode assignemnts
        potential_electrodes = np.setdiff1d(np.arange(number_of_electrode), unique)
        # the flattened coordinates array
        flat_coordinates = np.ravel_multi_index((self.y_coor, self.x_coor), f_interpolate[0].shape)
        # decide which one has noise and which doesn't by the amount of red -> larger than 0.75 in 0-1 scale:
        smearDictionary = {}
        for duplicate in duplicates:
            indices = np.where(order_electrode == duplicate)[0]
            flat_indices_sizes = []
            for index in indices:
                smearDictionary[index] = Viz_ICA_Streaming.Smear(index)
                thresh = 0.75
                array, size = Viz_ICA_Streaming._red_size(f_interpolate[index], thresh)
                smearDictionary[index].array = array
                smearDictionary[index].size = size
                flat_indices_sizes.append(size)
                smearDictionary[index].dup = duplicate

            not_noise = np.argmin(flat_indices_sizes)
            smearDictionary[indices[not_noise]].not_noise = 1

        combined_sorted_list, key_mapping_sorted = Viz_ICA_Streaming._combine_and_sort(smearDictionary)

        # loop from least noisy to most noisy
        for i, size in enumerate(combined_sorted_list):
            key = key_mapping_sorted[i]
            if (smearDictionary[key].not_noise != 1):  # if it is not a minumum from its duplicates
                array = smearDictionary[key].array
                noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                    0]]  # find the electrodes included in the red area
                thresh = 0.75
                overlap_bubble = False
                thresh_bool = False
                while (len(common_electrodes) == 0):
                    thresh = thresh - 0.05  # might make harsher - 0.75 - 0.5 - 0.25 steps
                    array, size = Viz_ICA_Streaming._red_size(f_interpolate[key], thresh)
                    noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                    common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                        0]]  # find the electrodes included in the red area

                    # if common_electrodes is empty, check if the area around electrode 15 is an option
                    if (len(common_electrodes) == 0 and 15 in potential_electrodes):
                        bubble_y, bubble_x = np.mgrid[self.y_coor[15]:self.y_coor[15] + 5, self.x_coor[15] - 4:self.x_coor[15] + 1]
                        electrode_15_bubble = np.ravel_multi_index((bubble_y, bubble_x), f_interpolate[0].shape)
                        overlap_bubble = np.any(np.in1d(electrode_15_bubble, array))
                        if (overlap_bubble):
                            chosen_electrode = 15
                            order_electrode[key] = chosen_electrode
                            break
                    # threshold to assign randomly and break if no source is found
                    if (thresh < 0.1):
                        chosen_electrode = potential_electrodes[0]
                        order_electrode[key] = chosen_electrode
                        thresh_bool = True
                        break

                if ((not overlap_bubble) and (thresh_bool == False)):
                    red_values = np.zeros(len(common_electrodes))
                    for j in range(len(common_electrodes)):
                        red_values[j] = f_interpolate[key][
                            self.y_coor[common_electrodes[j]], self.x_coor[common_electrodes[j]]]  # find the 'redness' value

                    chosen_electrode = common_electrodes[np.argmax(red_values)]  # assign the electrode that is most red
                    order_electrode[key] = chosen_electrode

                # update potential electrode list
                potential_electrodes = potential_electrodes[potential_electrodes != chosen_electrode]

        array_indices = np.argsort(order_electrode)[wanted_order]
        return array_indices, f_interpolate

    def _update_data(self, data: Data):

        # Get new data
        if self.plot_exg and data.exg_data is not None:
            n_exg_samples, n_exg_channels = data.exg_data.shape
            new_data_exg = data.exg_data[self.last_exg_sample:, :]
            self.last_exg_sample = n_exg_samples
        else:
            new_data_exg = []
        if self.plot_imu and data.imu_data is not None:
            n_imu_samples, n_imu_channels = data.imu_data.shape
            new_data_imu = data.imu_data[self.last_imu_sample:, :]
            self.last_imu_sample = n_imu_samples
        else:
            new_data_imu = []

        if self.plot_exg and self.plot_imu:
            q = data.fs_imu / data.fs_exg
            assert q - int(q) == 0
            q = int(q)
            new_data_imu = sig.resample_poly(new_data_imu, up=1, down=q)
            fs = data.fs_exg
        elif self.plot_exg:
            fs = data.fs_exg
        elif self.plot_imu:
            fs = data.fs_imu
        else:
            raise RuntimeError

        # Get old data
        old_data_exg = np.full((0, n_exg_channels), np.nan) if self.ydata is None else self.ydata[:, :n_exg_channels]
        old_data_imu = np.full((0, n_imu_channels), np.nan) if self.ydata is None else self.ydata[:, n_exg_channels:]
        old_data_exg = old_data_exg[~np.all(np.isnan(old_data_exg), axis=1), :]
        old_data_imu = old_data_imu[~np.all(np.isnan(old_data_imu), axis=1), :]

        data_exg = np.vstack((old_data_exg, new_data_exg)) if old_data_exg.size else new_data_exg
        data_imu = np.vstack((old_data_imu, new_data_imu)) if old_data_imu.size else new_data_imu

        if self.plot_exg and self.plot_imu:
            # Make full matrices same size and concatenate
            max_len = max(data_exg.shape[0], data_imu.shape[0])
            data_exg = Viz_ICA_Streaming._correct_matrix(data_exg, max_len)
            data_imu = Viz_ICA_Streaming._correct_matrix(data_imu, max_len)
            all_data = np.hstack((data_exg, data_imu))

        elif self.plot_exg:
            data_exg = Viz_ICA_Streaming._correct_matrix(data_exg, data_exg.shape[0])
            all_data = data_exg

        elif self.plot_imu:
            data_imu = Viz_ICA_Streaming._correct_matrix(data_imu, data_imu.shape[0])
            all_data = data_imu
        else:
            raise RuntimeError

        # Crop to correct size
        desired_samples = int(self.window_secs * fs)
        all_data = Viz_ICA_Streaming._correct_matrix(all_data, desired_samples)
        all_data, n_samples_cropped = Viz_ICA_Streaming._crop(all_data, desired_samples)
        if self.xdata is None:
            max_samples = max((n_imu_samples, n_exg_samples))
            last_sec = max_samples / fs
            ts_max = self.window_secs if max_samples <= self.window_secs * fs else last_sec
            ts_min = ts_max - self.window_secs
            self.xdata = np.arange(ts_min, ts_max, 1/fs)

        self.xdata += n_samples_cropped / fs
        self.ydata = all_data

    @staticmethod
    def _format_time(time):

        time_str = str(time)
        if '.' in time_str:
            time_str, ms = time_str.split('.')
            time_str = f'{time_str}.{ms[0]}'

        return time_str


    @staticmethod
    def _format_time_ICA(time):

        time_str = str(time)
        if '.' in time_str:
            sub_string=time_str.split('.')[0].split(':')
            if (sub_string[0] == '0'):#up to the first hour show min:sec only
                new_str = sub_string[1] + ':' + sub_string[2]
            else:
                new_str = sub_string[0] + ':' + sub_string[1] + ':' + sub_string[2]
        else:
            new_str='00:00'
        return new_str


    def filter_raw(self, y):
        # normalise cut-off frequencies to sampling frequency
        high_band = 35 / (self.fs / 2)
        low_band = 124 / (self.fs / 2)

        # create bandpass filter for EMG
        b1, a1 = butter(4, [high_band, low_band], btype='bandpass')

        # process EMG signal: filter EMG
        filtered_y = np.zeros(y.shape)
        for i in range(16):
            filtered_y[:, i] = filtfilt(b1, a1, y[:, i])

        # Design a notch filter to remove the 50 Hz power line interference
        f0 = 50  # Center frequency (Hz)
        Q = 30  # Quality factor
        w0 = f0 / (self.fs / 2)  # Normalized frequency
        b, a = iirnotch(w0, Q)

        # Apply the notch filter to the signal
        filtered_y = filtfilt(b, a, filtered_y)

        return filtered_y



    def get_ica_data(self, *args, **kwargs):
        #get data
        self._update_data(self.data)

        n_exg_channels = self.data.exg_data.shape[1] if self.plot_exg else 0
        n_imu_channels = self.data.imu_data.shape[1] if self.plot_imu else 0
        q = int(len(self.xdata) / self.max_points)
        n_pts = int(len(self.xdata) / q)
        x = sig.decimate(self.xdata, q)

        ynotnan = ~np.all(np.isnan(self.ydata), axis=1)
        y = self.ydata[ynotnan, :]
        # here I made a change and updated the resampling of 'y' with the customized ICA integration time
        # instead of n_pts
        y = y if np.any(np.isnan(y)) else sig.resample(y, int(self.ica_integration_time * self.fs), axis=0)

        for i in range(len(self.axes)):
            if(i%2==0):
                self.axes[i].set_xlim((x[0], x[-1]))

        if(self.filter_data==True):
            ica_y=self.filter_raw(y)
        else:
            ica_y=y

        # ica_start_time = time.time()
        K, W, Y = picard(ica_y[:, :n_exg_channels].T, n_components=16, ortho=True, max_iter=200)  # ICA algorithm
        # print('ICA took {} seconds'.format(time.time() - ica_start_time))  # to check the time it takes to run ICA

        inverse = np.absolute(inv(np.matmul(W, K)))
        grid_y, grid_x = np.mgrid[1:self.height + 1, 1:self.width + 1]
        points = np.column_stack((self.x_coor, self.y_coor))

        # find the order for the atlas
        array_indices, f_interpolate = Viz_ICA_Streaming.atlas(self, W, K, 16, self.wanted_order)


        self.f_interpolate=f_interpolate
        self.order=array_indices


        return f_interpolate, array_indices, Y, x, n_pts


    def update(self, *args, **kwargs):
        f_interpolate, order, model, x, n_pts = self.get_ica_data()

        source = 0
        for j in range(len(self.axes)):
            source = int(j / 2)
            if (j % 2 != 0):
                self.lines[j].set_array(f_interpolate[order[source]].ravel())
            else:
                new_model = sig.resample(model[order[source]], n_pts, axis=0)
                #self.lines[i].set_data(x,model[source])
                self.lines[j].set_data(x, new_model)


        # duration = how much time passed since starting - > this will be the text on the right
        # the time one the left will be calculated as the time that passed - the window time
        duration = datetime.now() - self.init_time
        time_right = Viz_ICA_Streaming._format_time_ICA(duration)
        time_left = max(timedelta(seconds=0), duration - timedelta(seconds=self.window_secs))
        time_left = Viz_ICA_Streaming._format_time_ICA(time_left)
        time_txt_artists = []
        for i in range(len(self.axes)):
            if(i%2==0):
                time_txt_artists.append(self.axes[i].text(0.1, 0.05, time_left, transform=self.axes[i].transAxes, ha="left", size=9))
                time_txt_artists.append(self.axes[i].text(0.85, 0.05, time_right, transform=self.axes[i].transAxes, ha="right", size=9))

        # Gather artists (necessary if using FuncAnimation)
        lines_artists = self.lines
        xtick_artists = []
        for i in range(len(self.axes)):
            for artist in self.axes[i].get_xticklabels():
                artist.axes = self.axes[i]
                xtick_artists.append(artist)
        #xtick_artists = [artist for ax in self.axes[-1, :] for artist in ax.get_xticklabels()]
        artists = lines_artists + time_txt_artists+  xtick_artists


        return artists

    def close(self, _):
        self.data.is_connected = False
        print('Window closed.')


    # Define the start/stop function for the button
    def start_stop_callback(self, event):
        if self.start_label == 'Start':
            self.start_label = 'Stop'
            self.start_time = datetime.now()
            self.button_start.label.set_text(self.start_label)
            self.timer_text.set_text('Elapsed time: 0.00 seconds')
            self.timer.start()  # Start the timer
            self.timer_running = True
        else:
            self.start_label = 'Start'
            elapsed_time = datetime.now() - self.start_time
            print('Elapsed time:', elapsed_time.total_seconds())
            self.ica_integration_time = elapsed_time.total_seconds()
            print('ICA integration time:', self.ica_integration_time)
            self.button_start.label.set_text(self.start_label)
            self.timer.stop()  # Stop the timer
            self.timer_running = False

    def update_timer(self):
        if self.timer_running:
            elapsed_time = datetime.now() - self.start_time
            self.timer_text.set_text('Elapsed time: {:.2f} seconds'.format(elapsed_time.total_seconds()))
            self.figure.canvas.draw_idle()

    def start_animation(self,event):
        self.animation.event_source.start()

    # to stop the animation
    def stop_animation(self, event):
        self.animation.event_source.stop()

    # to edit the ICA integration time
    def submit_integ_time(self, text):
        try:
            self.ica_integration_time = float(text)
            # self.window_secs = float(text)
            # Use the entered number in your script as needed
        except ValueError:
            print("Invalid input. Please enter a number.")

        # print(self.window_secs)
        print(self.ica_integration_time)

    def start(self):
        # do  blit = False to change the xaxis length....
        self.animation = FuncAnimation(self.figure, self.update,
                                  blit=True, interval=self.update_interval_ms, repeat=False, cache_frame_data=False)

        # create a timer object
        self.timer = self.figure.canvas.new_timer(interval=200)
        # add callback to timer
        self.timer.add_callback(self.update_timer)

        # Create a separate axes for the timer text
        self.timer_ax = self.figure.add_axes([0.11, 0.01, 0.05, 0.025])
        self.timer_ax.axis('off')  # Turn off the axes
        # create text object which will be updated every 0.1 second
        self.timer_text = self.timer_ax.text(0.11, 0.01, 'Elapsed time: 0.00 seconds', transform=self.timer_ax.transAxes, ha="left", va="center")

        # Create start and stop buttons
        ax_start = plt.axes([0.05, 0.01, 0.05, 0.025])
        self.button_start = Button(ax_start, self.start_label)
        self.button_start.on_clicked(self.start_stop_callback)

        # ax_stop = plt.axes([0.11, 0.01, 0.05, 0.025])
        # button_stop = Button(ax_stop, 'Stop')
        # button_stop.on_clicked(self.stop_animation)

        textbox_ax = plt.axes([0.37, 0.01, 0.07, 0.025])
        integ_time = TextBox(textbox_ax, "ICA integration time:", initial='10')
        integ_time.on_submit(self.submit_integ_time)

        self.animation._start()
        # plt.show()
