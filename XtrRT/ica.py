from XtrRT.viz import Viz

import warnings
import sklearn.exceptions
from sklearn.decomposition import FastICA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import ImageGrab
import numpy as np
from picard import picard #*need to cite*,  install "python-picard"
from numpy.linalg import inv
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from scipy.signal import butter, lfilter, filtfilt, iirnotch #for filtering the data



from .data import Data, ConnectionTimeoutError


#for user inputs (GUI)
import tkinter as tk

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Viz_ICA:

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
        self.all_data=None
        self.lines = []
        self.bg = None
        self.last_exg_sample = 0
        self.last_imu_sample = 0
        self.init_time = datetime.now()
        self.update_interval_ms = update_interval_ms
        self.max_points = max_points
        self.find_emg = find_emg
        self.filters = filters
        self._backend = None
        self._existing_patches = []

        #for interactive ICA
        self.times={}
        self.button_pressed = False
        self.offset_before=None
        self.offset_after=None
        self.sampling_rate=None
        self.checkboxes=[]
        self.wanted_order=np.array([12, 13, 14, 15,
                             11, 10, 9, 8,
                             4, 5, 6, 7,
                             3, 2, 1, 0])
        self.y_coor=y_coor
        self.x_coor=x_coor
        self.width=width
        self.height=height
        self.grid_x=None
        self.grid_y=None
        self.points=None
        self.image=image
        self.n_imu_channels=None
        self.n_exg_channels=None
        self.n_imu_samples=None
        self.filter_data = filter_data

        # Confirm initial data retrieval before continuing (or raise Error if timeout)
        while not (self.data.is_connected and self.data.has_data):
            plt.pause(0.01)
            if max_timeout is not None and (datetime.now() - self.init_time).seconds > max_timeout:
                if not data.is_connected:
                    raise ConnectionTimeoutError
                elif not data.has_data:
                    raise TimeoutError(f"Did not succeed to stream data within {max_timeout} seconds.")


        #set the sampling_rate
        self.sampling_rate = self.data.fs_exg
        #set the grid_x, grid_y, and points
        self.grid_y, self.grid_x = np.mgrid[1:self.height + 1, 1:self.width + 1]
        self.points = np.column_stack((self.x_coor, self.y_coor))
        # Get data
        n_exg_samples, self.n_exg_channels = self.data.exg_data.shape if self.plot_exg else (0, 0)  # TODO: can be None if imu data comes first
        self.n_imu_samples, self.n_imu_channels = self.data.imu_data.shape if self.plot_imu else (0, 0)  # TODO: can be None if exg data comes first
        # Make timestamp vector
        max_samples = max((self.n_imu_samples, n_exg_samples))
        last_sec = max_samples / self.sampling_rate
        ts_max = self.window_secs if max_samples <= self.window_secs * self.sampling_rate else last_sec
        ts_min = ts_max - self.window_secs
        ts = np.arange(ts_min, ts_max, 1 / self.sampling_rate)
        if self.max_points is None:
            self.max_points = len(ts)
        else:
            ts = Viz_ICA._downsample_monotonic(ts, n_pts=self.max_points)

        #
        n_channels = self.n_exg_channels + self.n_imu_channels
        self.xdata = ts
        self.ydata = np.full((len(ts), n_channels), np.nan)


    @staticmethod
    def _downsample_monotonic(arr: np.ndarray, n_pts: int):
        """
        Returns values of `arr` at closest indices to every n_pts_new/arr.shape[0]

        :param arr: 1D array
        :param n_pts_new: number of desired points
        :return: downsampled array of shape (n_pts,)
        """

        out = arr[[int(np.round(idx)) for idx in np.linspace(0, len(arr)-1, n_pts)]]

        return out

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

    def _update_data(self, data: Data):

        # Get new data
        if self.plot_exg and data.exg_data is not None:
            n_exg_samples, n_exg_channels = data.exg_data.shape
            new_data_exg = data.exg_data[self.last_exg_sample:, :]
            self.last_exg_sample = n_exg_samples
        else:
            new_data_exg = []
        if self.plot_imu and data.imu_data is not None:
            self.n_imu_samples, self.n_imu_channels = data.imu_data.shape
            new_data_imu = data.imu_data[self.last_imu_sample:, :]
            self.last_imu_sample = self.n_imu_samples
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
        old_data_imu = np.full((0, self.n_imu_channels), np.nan) if self.ydata is None else self.ydata[:, n_exg_channels:]

        data_exg = np.vstack((old_data_exg, new_data_exg)) if old_data_exg.size else new_data_exg
        data_imu = np.vstack((old_data_imu, new_data_imu)) if old_data_imu.size else new_data_imu

        if self.plot_exg and self.plot_imu:
            # Make full matrices same size and concatenate
            max_len = max(data_exg.shape[0], data_imu.shape[0])
            data_exg = Viz_ICA._correct_matrix(data_exg, max_len)
            data_imu = Viz_ICA._correct_matrix(data_imu, max_len)
            all_data = np.hstack((data_exg, data_imu))

        elif self.plot_exg:
            data_exg = Viz_ICA._correct_matrix(data_exg, data_exg.shape[0])
            all_data = data_exg

        elif self.plot_imu:
            data_imu = Viz_ICA._correct_matrix(data_imu, data_imu.shape[0])
            all_data = data_imu
        else:
            raise RuntimeError


        ts_max = self.window_secs
        ts_min = ts_max - self.window_secs
        self.xdata = np.arange(ts_min, ts_max, 1/fs)

        #not cropped!
        self.all_data = all_data

    @staticmethod
    def _format_time(time):

        time_str = str(time)
        if '.' in time_str:
            time_str, ms = time_str.split('.')
            time_str = f'{time_str}.{ms[0]}'

        return time_str

    @staticmethod
    def _nandecimate(data: np.ndarray, n_pts: int):
        """
        NaN-safe decimation.
        Removes NaNs, decimates NaN-removed data, then re-inserts NaNs at best-estimated indices.

        :param data: 2D array (nsamples, nchannels)
        :param n_pts: number of desired data points
        :return: decimated data (with NaNs) of shape (n_pts, nchannels)
        """

        if data.shape[0] <= n_pts:
            return data

        non_nan_idc = np.where(~np.all(np.isnan(data), axis=1))[0]
        data_nonan = data[non_nan_idc, :]
        proportion_nonan = data_nonan.shape[0] / data.shape[0]
        n_pts_resample = int(np.round(n_pts * proportion_nonan))
        resampled = sig.resample(data_nonan, n_pts_resample, axis=0)
        out = np.full((n_pts, data.shape[1]), np.nan)
        non_nan_idc_downsampled = np.array([int(i * n_pts / data.shape[0]) for i in
                                            non_nan_idc[np.array([int(np.round(idx)) for idx in
                                                                  np.linspace(0, len(non_nan_idc) - 1, num=n_pts_resample)])]])

        # To avoid inserting at index n_pts, shift all indices after the last nan down by 1
        if n_pts in non_nan_idc_downsampled:
            nan_idc_downsampled = [idx for idx in np.arange(0, n_pts) if idx not in non_nan_idc_downsampled]
            last_nan_idx = nan_idc_downsampled[-1]
            non_nan_idc_downsampled = np.array([idx if idx < last_nan_idx else idx - 1 for idx in non_nan_idc_downsampled])

        out[non_nan_idc_downsampled, :] = resampled

        return out

    @staticmethod
    def _red_size(arr, thresh):
        myrange = np.nanmax(arr) - np.nanmin(arr)
        norm_arr = (arr - np.nanmin(arr)) / myrange
        # threshold for red/orange 0.75-0.1
        condition = (norm_arr.ravel()) > thresh
        flat_indices = np.where(condition)[0]

        return flat_indices, flat_indices.shape[0]

    @staticmethod
    def _norm(arr):
        myrange = np.nanmax(arr) - np.nanmin(arr)
        norm_arr = (arr - np.nanmin(arr)) / myrange
        return norm_arr


    def _atlas_duplicates_sig(self, duplicates, potential_electrodes, order_electrode, f_interpolate, flat_coordinates, sig):
        for duplicate in duplicates:
            flat_indices = []
            flat_indices_sizes = []
            condition = lambda x: x == duplicate
            indices = [index for index, element in enumerate(order_electrode) if index in sig and condition(element)]
            for index in indices:
                thresh = 0.75
                array, size = Viz_ICA._red_size(f_interpolate[index], thresh)
                flat_indices.append(array)
                flat_indices_sizes.append(size)
            # change the electrode number assignment for those that aren't min
            not_noise = np.argmin(flat_indices_sizes)
            for i in range(len(flat_indices)):
                if (i != not_noise):
                    index = indices[i]
                    array = flat_indices[i]
                    noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                    common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                        0]]  # find the electrodes included in the red area
                    # common_electrodes=[el for el in common_electrodes if el not in duplicates] #exlude duplicates from list
                    while (len(common_electrodes) == 0):
                        thresh = thresh - 0.05  # might make harsher - 0.75 - 0.5 - 0.25 steps
                        array, size = Viz_ICA._red_size(f_interpolate[index], thresh)
                        noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                        common_electrodes = noise_electrodes[
                            np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                                0]]  # find the electrodes included in the red area
                    red_values = np.zeros(len(common_electrodes))
                    for j in range(len(common_electrodes)):
                        red_values[j] = f_interpolate[index][
                            self.y_coor[common_electrodes[j]], self.x_coor[common_electrodes[j]]]  # find the 'redness' value

                    chosen_electrode = common_electrodes[np.argmax(red_values)]  # assign the electrode that is most red
                    order_electrode[indices[i]] = chosen_electrode

                    # update potential electrode list
                    potential_electrodes = potential_electrodes[potential_electrodes != chosen_electrode]

        return potential_electrodes, order_electrode


    def _atlas_duplicates_not_sig(self, duplicates, potential_electrodes, order_electrode, f_interpolate, flat_coordinates,
                                 sig):
        for duplicate in duplicates:
            flat_indices = []
            flat_indices_sizes = []
            condition = lambda x: x == duplicate
            indices = [index for index, element in enumerate(order_electrode) if condition(element)]
            for index in indices:
                thresh = 0.75
                array, size = Viz_ICA._red_size(f_interpolate[index], thresh)
                flat_indices.append(array)
                flat_indices_sizes.append(size)

            common_values = np.intersect1d(indices, sig)

            if common_values.size == 0:  # if sig is not included in index, chose min area as the non_noise channel
                not_noise = np.argmin(flat_indices_sizes)
            else:  # otherwise, the significant channel is the not noise channel
                not_noise = np.where(np.isin(indices, common_values))[0][0]

            for i in range(len(flat_indices)):
                thresh = 0.75
                if (i != not_noise):
                    index = indices[i]
                    array = flat_indices[i]
                    noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                    common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                        0]]  # find the electrodes included in the red area
                    while (len(common_electrodes) == 0):
                        thresh = thresh - 0.05  # might make harsher - 0.75 - 0.5 - 0.25 steps
                        if (thresh <= 0.5):
                            chosen_electrode = -1  # assign -1: noisy channel that would be put aside
                            break
                        array, size = Viz_ICA._red_size(f_interpolate[index], thresh)
                        noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                        common_electrodes = noise_electrodes[
                            np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                                0]]  # find the electrodes included in the red area
                    if (thresh > 0.5):
                        red_values = np.zeros(len(common_electrodes))
                        for j in range(len(common_electrodes)):
                            red_values[j] = f_interpolate[index][
                                self.y_coor[common_electrodes[j]], self.x_coor[common_electrodes[j]]]  # find the 'redness' value

                        chosen_electrode = common_electrodes[
                            np.argmax(red_values)]  # assign the electrode that is most red
                    order_electrode[indices[i]] = chosen_electrode

                    # update potential electrode list
                    potential_electrodes = potential_electrodes[potential_electrodes != chosen_electrode]

        return potential_electrodes, order_electrode

    def _calculate_heatmaps(self,relative_intensity, sort_intensity, W, K):
        # calculates heatmaps and finds inital guesses for the electrodes

        # threshold for significance 1.8 (after rounding)
        number_of_sig = len(np.where(np.round(relative_intensity[sort_intensity], 2) >= 1.8)[0])
        sig = sort_intensity[-number_of_sig:]

        inverse = np.absolute(inv(np.matmul(W, K)))
        f_interpolate = []
        # plot the 16 electrodes in a way that makes sense
        order_electrode = np.full((16,), -1)

        for i in range(16):  #
            # f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))
            interpolate_data = griddata(self.points, inverse[:, i], (self.grid_x, self.grid_y), method='linear')

            norm_arr = Viz_ICA._norm(interpolate_data)
            f_interpolate.append(norm_arr)

            # if i in sig:
            ## find red point
            electrode = np.argmax(inverse[:, i])
            order_electrode[i] = electrode

        return f_interpolate, order_electrode, sig

    def _calculate_atlas(self, order_electrode, f_interpolate, sig):
        # calculates order of atlas and takes care of duplicates/noises
        # currently, for real time, there aren't always 16 clear sources, after a certain threshold they are assigned '-1' and might not show

        # the potential electrode assignemnts
        unique = np.unique(order_electrode)
        potential_electrodes = np.setdiff1d(np.arange(16), unique)

        # the flattened coordinates array
        flat_coordinates = np.ravel_multi_index((self.y_coor, self.x_coor), f_interpolate[0].shape)

        # go over duplicates of electrodes assignments and decide which one has noise and which doesn't by the amount of red
        # start with threshold of 0.75 and go down if nececary

        # priority for the signifcant electrodes
        unique, counts = np.unique(order_electrode[sig], return_counts=True)
        duplicates = unique[counts > 1]

        potential_electrodes, order_electrode = self._atlas_duplicates_sig(duplicates, potential_electrodes, order_electrode,
                                                                     f_interpolate, flat_coordinates, sig)

        # next do the same procedure for all other electrodes
        unique, counts = np.unique(order_electrode, return_counts=True)
        duplicates = unique[counts > 1]

        potential_electrodes, order_electrode = self._atlas_duplicates_not_sig(duplicates, potential_electrodes,
                                                                         order_electrode, f_interpolate,
                                                                         flat_coordinates,
                                                                         sig)

        # insert the potential electrodes in the order_electrodes array just to order it correctly
        order_electrode[np.where(order_electrode == -1)[0]] = potential_electrodes

        array_indices = np.argsort(order_electrode)[self.wanted_order]

        return array_indices, order_electrode, potential_electrodes

    # Function to handle checkbox selection
    def _update_visibility(self, axs, index):
        visibility = axs[index].get_visible()  # Get the current visibility state of the first subplot
        axs[index].set_visible(not visibility)  # Toggle the visibility of the first subplot (the signal)
        axs[index + 1].set_visible(not visibility)  # Toggle the visibility of the subplot next to it (the heatmap)


        plt.draw()

    def _plot_sources(self, window, potential_electrodes, f_interpolate, array_indices, start_time, end_time, Y, sig_convert):
        # heatmaps with sources
        row_num, col_num = 4, 12
        ratio = round(self.height / self.width, 1)

        fig, axes = plt.subplots()
        spec = fig.add_gridspec(row_num, col_num)
        # fig.set_size_inches(1 * col_num, 1 * ratio * row_num)
        # Autoscale the figure to fit the screen
        fig.set_dpi(300)

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
                axs[j].pcolormesh(f_interpolate[array_indices[source]], cmap='jet', alpha=0.5)
                axs[j].set_aspect('auto')
                axs[j].axis('off')
            else:
                duration = end_time - start_time
                axs[j].plot(np.arange(start_time, end_time), Y[array_indices[source], -duration:],
                            color='tab:blue', lw=0.5)
                axs[j].margins(0)
                axs[j].set_ylim([-5, 5])
                axs[j].set_title('source %d' % (self.wanted_order[source] + 1), fontsize=7)

            # only show the y-axis for the left-most column
            if (j % 8 == 0):
                axs[j].tick_params(axis='y', labelsize=5, direction='in', length=4, width=1)
            else:
                axs[j].tick_params(axis='y', left=False, labelleft=False)

            # only show the x-axis for the bottom-most column
            if (j >= 24):
                axs[j].tick_params(axis='x', labelsize=5, direction='in', length=4, width=1)
            else:
                axs[j].tick_params(axis='x', bottom=False, labelbottom=False)

            time_func = (lambda a: divmod(int(a / self.sampling_rate), 60))
            ticks = ticker.FuncFormatter(
                lambda x, pos: (str(time_func(x)[0]).zfill(2) + ':' + str(time_func(x)[1]).zfill(2)))
            axs[j].xaxis.set_major_formatter(ticks)

        global checkboxes, checkbox_vars

        # Create a list to store the IntVar variables for checkbox states
        checkbox_vars = []

        # Create checkboxes for each subplot
        for i in range(0, len(axs), 2):
            source = int(i / 2)

            var = tk.IntVar()

            if (self.wanted_order[source] in sig_convert):  # most significant subplots are selected by default
                var.set(1)
                axs[i].set_visible(True)
                axs[i + 1].set_visible(True)
            else:
                var.set(0)
                axs[i].set_visible(False)
                axs[i + 1].set_visible(False)

            checkbox = tk.Checkbutton(window, text='Source %d' % (self.wanted_order[source] + 1), variable=var,
                                      command=lambda i=i: self._update_visibility(axs, i))
            # checkbox.select()  # To initially select all checkboxes uncomment this line

            # Customize the appearance of the checkbox text
            if (self.wanted_order[source] in sig_convert):
                checkbox.config(
                    bg='yellow',  # Set yellow background
                    fg='black',  # Set font color
                    font=('Arial', 12, 'bold')  # Set font style (Arial, size 12, bold)
                )

            if (self.wanted_order[source] in potential_electrodes):
                checkbox.config(
                    fg='grey',  # Set grey font color
                    font=('Arial', 5)  # Set font style
                )

            checkbox.grid(row=int(source / 4), column=source % 4)

            checkbox_vars.append(var)
            self.checkboxes.append(checkbox)


        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed')
        fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.95, hspace=0.5)

        plt.show()



    def filter_raw(self, y):
        # normalise cut-off frequencies to sampling frequency
        high_band = 35 / (self.sampling_rate / 2)
        low_band = 124 / (self.sampling_rate / 2)

        # create bandpass filter for EMG
        b1, a1 = butter(4, [high_band, low_band], btype='bandpass')

        # process EMG signal: filter EMG
        filtered_y = np.zeros(y.shape)
        for i in range(16):
            filtered_y[:, i] = filtfilt(b1, a1, y[:, i])

        # Design a notch filter to remove the 50 Hz power line interference
        f0 = 50  # Center frequency (Hz)
        Q = 30  # Quality factor
        w0 = f0 / (self.sampling_rate / 2)  # Normalized frequency
        b, a = iirnotch(w0, Q)

        # Apply the notch filter to the signal
        filtered_y = filtfilt(b, a, filtered_y)

        return filtered_y

    def _obtain_sources_for_segement(self, ica_time=None):
        #define the offsets before and after the expression (to see the change in the magnitude)
        ob = int(self.offset_before * self.sampling_rate)
        oa = int(self.offset_after * self.sampling_rate)

        #obtain the start and end times
        time_from_recording=(self.times["Start Expression"]-self.init_time).total_seconds()
        start_time = int(time_from_recording * self.sampling_rate) - ob

        time_from_recording = (self.times["End Expression - Relax Again"] - self.init_time).total_seconds()
        end_time = int(time_from_recording * self.sampling_rate) + oa

        #calculate the duration. The length of the plotted segements are defined by the duration.
        duration = end_time - start_time

        #the segement taken into account for the ICA is defined by 'windows_sec'
        #either by the duration time (defined by the expression) or by ica_time defined by the user
        if (not ica_time):
            self.window_secs= int(duration/self.sampling_rate)
            desired_samples=int(duration)
        else:
            self.window_secs=ica_time
            desired_samples=int(ica_time*self.sampling_rate)

        # update timestamp vector
        ts_max = self.window_secs
        ts_min = ts_max - self.window_secs
        ts = np.arange(ts_min, ts_max, 1/self.sampling_rate)
        self.max_points = len(ts)

        #get the updated data until the expression ended (+relax)
        #update data without cropping
        if (not ica_time):
            self._update_data(self.data)

        # Crop to desired size - determined by desired_samples defined in the if condition above
        #either from initial relax to final relax or 'take back' ica_time seconds from final relax
        self.all_data = Viz_ICA._correct_matrix(self.all_data, desired_samples)
        self.all_data, n_samples_cropped = Viz_ICA._crop(self.all_data, desired_samples)
        self.ydata=self.all_data

        #process the data
        n_exg_channels = self.data.exg_data.shape[1] if self.plot_exg else 0
        self.n_imu_channels = self.data.imu_data.shape[1] if self.plot_imu else 0
        q = int(len(self.ydata) / self.max_points)
        n_pts = int(len(self.ydata) / q)
        y = Viz_ICA._nandecimate(self.ydata, n_pts)
        non_nan_idc = np.where(~np.all(np.isnan(self.ydata[:, :n_exg_channels]), axis=1))[0]
        y = self.ydata[non_nan_idc, :n_exg_channels]
        sigbufs=y[:,:n_exg_channels]

        if (self.filter_data == True):
            ica_y = self.filter_raw(sigbufs)
        else:
            ica_y = sigbufs

        # run the ICA algorithm
        K, W, Y = picard(ica_y.T, n_components=16, ortho=True,
                         max_iter=200)

        #find the signal of the sources during the relax period and during the expression (for comparison of magnitude)
        relax_calib_per_expression = Y[:, -duration:-duration + ob]
        seg_sources = Y[:, -duration + ob:-oa]
        relative_intensity = np.average(np.abs(seg_sources), axis=1) / np.average(np.abs(relax_calib_per_expression),
                                                                                  axis=1)
        sort_intensity = np.argsort(relative_intensity) #order by intensity (least to most) 

        return Y, K, W, sort_intensity, relative_intensity, start_time, end_time



    def close(self, _):
        self.data.is_connected = False
        print('Window closed.')


    #when specifiying the specific ica_time
    def get_time(self):
        time_value = int(self.entry_time.get())
        ica_time = int(self.entry_time.get()) * self.sampling_rate

        # Clear previous checkboxes
        for checkbox in self.checkboxes:
            checkbox.destroy()

        # Perform any desired actions with the input value
        Y, K, W, sort_intensity, relative_intensity, start_time, end_time = self._obtain_sources_for_segement(ica_time)
        f_interpolate, order_electrode, sig = self._calculate_heatmaps(relative_intensity, sort_intensity, W, K)
        #array_indices, order_electrode, potential_electrodes = self._calculate_atlas(order_electrode, f_interpolate,  sig)
        # for gibbrish data 'calculate_atlas' gets stuck in a loop  - insert fake data when debugging
        array_indices = np.arange(16)
        order_electrode = np.arange(16)
        potential_electrodes = np.arange(4)

        sig_convert = self.wanted_order[np.where(np.isin(array_indices, sig))[0]]
        self._plot_sources(self.window, potential_electrodes, f_interpolate, array_indices, start_time, end_time, Y,
                           sig_convert)

    def button_click(self,text):
        self.times[text] = datetime.now()
        if (text == "Start Expression"):
            self.data.add_annotation("Start Expression")
            self.offset_before = (self.times["Start Expression"] - self.times["Start Relax"]).total_seconds()
        elif (text == "End Relax"):
            self.offset_after = (self.times["End Relax"] - self.times["End Expression - Relax Again"]).total_seconds()
        elif(text=='Start Relax'):
            self.data.add_annotation("Start Relax")

        # after clicking 'End Relax' (final relax) kick-start the real-time data collection + analysis
        if (text == "End Relax"):
            if self.button_pressed:
                # Clear previous checkboxes
                for checkbox in self.checkboxes:
                    checkbox.destroy()
            else:
                # print("Button was pressed for the first time.")
                self.button_pressed = True


            # Perform any desired actions with the input value
            Y, K, W, sort_intensity, relative_intensity, start_time, end_time = self._obtain_sources_for_segement()
            f_interpolate, order_electrode, sig = self._calculate_heatmaps(relative_intensity, sort_intensity, W, K)
            #for real data
            array_indices, order_electrode, potential_electrodes = self._calculate_atlas(order_electrode, f_interpolate, sig)
            # for gibbrish data 'calculate_atlas' gets stuck in a loop  - insert fake data when debugging
            #array_indices = np.arange(16)
            #order_electrode = np.arange(16)
            #potential_electrodes = np.arange(4)

            sig_convert = self.wanted_order[np.where(np.isin(array_indices, sig))[0]]
            self._plot_sources(self.window, potential_electrodes, f_interpolate, array_indices, start_time, end_time, Y,
                               sig_convert)

    def start(self):
        # Create the main window
        self.window = tk.Tk()

        #indicator of whether 'end relax' was pressed for the first time
        self.button_pressed = False

        button_texts = ["Start Relax", "Start Expression", "End Expression - Relax Again", "End Relax"]

        for index, text in enumerate(button_texts):
            button = tk.Button(self.window, text=text, command=lambda t=text: self.button_click(t))
            button.grid(row=7 + index, column=0)

        # Create an entry field - for accumulation time
        self.entry_time = tk.Entry(self.window)
        self.entry_time.grid(row=11, column=0)

        # Create a button to get the input value
        button = tk.Button(self.window, text="Submit", command=self.get_time)
        button.grid(row=12, column=0)

        # Start the Tkinter event loop
        self.window.mainloop()





