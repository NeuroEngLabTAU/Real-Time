
import warnings
import sklearn.exceptions
from sklearn.decomposition import FastICA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import ImageGrab
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, iirnotch #for filtering the data
from matplotlib.widgets import Button #for button in funcanimator

import time

from .data import Data, ConnectionTimeoutError

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Viz_spec:

    def __init__(self,
                 data: Data,
                 window_secs: float = 10,
                 plot_exg: bool = True,
                 plot_imu: bool = True,
                 plot_spectogram: bool = True,
                 ylim_exg: tuple = (-350, 350),
                 ylim_imu: tuple = (-1, 1),
                 update_interval_ms: int = 200,
                 max_points: (int, None) = 1000,
                 max_timeout: (int, None) = 15,
                 find_emg: bool = False,
                 filters: dict = None,
                 filter_data: bool=False,
                 figure=None, ):

        assert plot_exg or plot_imu

        self.data = data
        self.plot_exg = plot_exg
        self.plot_imu = plot_imu
        self.plot_spectogram = plot_spectogram if plot_spectogram and plot_exg else False
        self.window_secs = window_secs
        self.axes = None
        self.figure = figure
        self.ylim_exg = ylim_exg
        self.ylim_imu = ylim_imu
        self.xdata = None
        self.ydata = None
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
        self.filter_data = filter_data
        self.fs = None

        # Confirm initial data retrieval before continuing (or raise Error if timeout)
        while not (self.data.is_connected and self.data.has_data):
            plt.pause(0.01)
            if max_timeout is not None and (datetime.now() - self.init_time).seconds > max_timeout:
                if not data.is_connected:
                    raise ConnectionTimeoutError
                elif not data.has_data:
                    raise TimeoutError(f"Did not succeed to stream data within {max_timeout} seconds.")

        self.setup()

    def setup(self):

        self.ylim_imu = self.ylim_imu if self.ylim_imu else self.ylim_imu
        self.ylim_exg = self.ylim_exg if self.ylim_exg else self.ylim_exg

        # Get data
        n_exg_samples, n_exg_channels = self.data.exg_data.shape if self.plot_exg else (0, 0)  # TODO: can be None if imu data comes first
        n_imu_samples, n_imu_channels = self.data.imu_data.shape if self.plot_imu else (0, 0)  # TODO: can be None if exg data comes first
        self.fs = self.data.fs_exg if self.plot_exg else self.data.fs_imu

        # Make timestamp vector
        max_samples = max((n_imu_samples, n_exg_samples))
        last_sec = max_samples / self.fs
        ts_max = self.window_secs if max_samples <= self.window_secs*self.fs else last_sec
        ts_min = ts_max - self.window_secs
        ts = np.arange(ts_min, ts_max, 1/self.fs)
        if self.max_points is None:
            self.max_points = len(ts)
        else:
            ts = Viz_spec._downsample_monotonic(ts, n_pts=self.max_points)

        #
        n_channels = n_exg_channels + n_imu_channels
        self.xdata = ts
        self.ydata = np.full((len(ts), n_channels), np.nan)

        # For auto-maximization of figure
        screensize = ImageGrab.grab().size
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        screensize_inches = [px*npx for npx in screensize]

        # Prepare plots
        n_rows = n_exg_channels + 1*self.plot_imu
        n_cols = 2 if self.plot_spectogram else 1
        f, axs = plt.subplots(n_rows, n_cols, sharex=False, figsize=screensize_inches)
        axs = np.atleast_2d(axs).T if n_cols == 1 else axs
        plt.subplots_adjust(hspace=0, left=0.07, right=0.98, top=0.96, bottom=0.04)
        for col in range(n_cols):
            for row in range(n_exg_channels):
                ax = axs[row, col]
                line, = ax.plot(self.xdata, self.ydata[:, row], linewidth=0.5)
                ax.set_ylim((-1, 1)) if col == 1 else axs[row, col].set_ylim(self.ylim_exg)
                self.lines.append(line)
                ax.set_ylabel(f"{'IC' if col == 1 else 'col'} {row+1 if col == 1 else row}", fontsize=11-np.sqrt(n_rows))
                ax.xaxis.set_ticklabels([])
            for n, ch in enumerate(range(n_imu_channels)):
                line, = axs[-1, col].plot(self.xdata, self.ydata[:, n_exg_channels + n], linewidth=0.5)
                self.lines.append(line)
                axs[-1, col].set_ylabel("IMU") if col == 0 else None
                axs[-1, col].set_ylim(*self.ylim_imu)
        axs[0, 0].set_xlim((ts[0], ts[-1]))
        [axs[row, col].spines['bottom'].set_visible(False) for row in range(n_rows-1) for col in range(n_cols)]
        [axs[row, col].spines['top'].set_visible(False) for row in range(1, n_rows) for col in range(n_cols)]
        axs[0, 0].set_title("Raw data")
        axs[0, 1].set_title("ICA") if self.plot_spectogram else None
        # self._timer = axs[-1, -1].text(1, 0.05, "", transform=axs[-1, -1].transAxes, ha="right", size=9)

        for ax in axs[-1, :]:
            ax.tick_params(axis='x',          # changes apply to the x-axis
                           which='both',      # both major and minor ticks are affected
                           bottom=False,      # ticks along the bottom edge are off
                           top=False,         # ticks along the top edge are off
                           labelbottom=True)  # labels along the bottom edge are off

        self.axes = axs
        self.figure = f
        self.figure.canvas.mpl_connect('close_event', self.close)

        self.bg = [self.figure.canvas.copy_from_bbox(ax.bbox) for ax in np.ravel(self.axes)] if self._backend != 'module://mplopengl.backend_qtgl' else None

    # def maximize(self):
    #
    #     # plt.figure(self.figure)
    #     self._backend = matplotlib.get_backend()
    #     mng = plt.get_current_fig_manager()
    #
    #     if self._backend == 'TkAgg':
    #         mng.window.state('zoomed')
    #     elif self._backend == 'wxAgg':
    #         mng.frame.Maximize(True)
    #     elif self._backend in ('QtAgg', 'Qt4Agg'):
    #         mng.window.showMaximized()
    #     else:
    #         print(f"Don't know how to maximize a {self._backend} figure")

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

        data_exg = np.vstack((old_data_exg, new_data_exg)) if old_data_exg.size else new_data_exg
        data_imu = np.vstack((old_data_imu, new_data_imu)) if old_data_imu.size else new_data_imu

        if self.plot_exg and self.plot_imu:
            # Make full matrices same size and concatenate
            max_len = max(data_exg.shape[0], data_imu.shape[0])
            data_exg = Viz_spec._correct_matrix(data_exg, max_len)
            data_imu = Viz_spec._correct_matrix(data_imu, max_len)
            all_data = np.hstack((data_exg, data_imu))

        elif self.plot_exg:
            data_exg = Viz_spec._correct_matrix(data_exg, data_exg.shape[0])
            all_data = data_exg

        elif self.plot_imu:
            data_imu = Viz_spec._correct_matrix(data_imu, data_imu.shape[0])
            all_data = data_imu
        else:
            raise RuntimeError

        # Crop to correct size
        desired_samples = int(self.window_secs * fs)
        all_data = Viz_spec._correct_matrix(all_data, desired_samples)
        all_data, n_samples_cropped = Viz_spec._crop(all_data, desired_samples)

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

    def update(self, *args, **kwargs):

        self._update_data(self.data)

        n_exg_channels = self.data.exg_data.shape[1] if self.plot_exg else 0
        n_imu_channels = self.data.imu_data.shape[1] if self.plot_imu else 0
        q = int(len(self.xdata)/self.max_points)
        n_pts = int(len(self.xdata) / q)
        x = sig.decimate(self.xdata, q)
        ynotnan = ~np.all(np.isnan(self.ydata), axis=1)
        y = self.ydata[ynotnan, :]
        y = y if np.any(np.isnan(y)) else sig.resample(y, n_pts, axis=0)

        # if (self.filter_data == True):
        #     viz_y = self.filter_raw(y)
        # else:
        viz_y = y

        for n in range(n_exg_channels + n_imu_channels):
            self.lines[n].set_data(x, viz_y[:, n])
            self.axes[n, 0].set_xlim((x[0], x[-1]))  # Set xlim for time-domain plot

        # self.axes[-1, -1].set_xlim((x[0], x[-1]))

        # Plot Spectogram (if relevant)
        if self.plot_spectogram:
            from scipy.signal import get_window
            window_size = int(self.window_secs * self.data.fs_exg)  # Window size: 500 ms

            for n in range(n_exg_channels):
                # Define the window function
                window_function = get_window('hann', window_size)
                windowed_segment = viz_y[:, n] * window_function

                # Compute the FFT of the windowed segment
                fft_result = np.fft.fft(windowed_segment)
                fft_magnitude = np.abs(fft_result)
                frequencies = np.fft.fftfreq(len(fft_magnitude), 1 / self.data.fs_exg)

                # Only take the positive frequencies and corresponding magnitudes
                positive_frequencies = frequencies[:len(frequencies) // 2]
                positive_magnitude = fft_magnitude[:len(fft_magnitude) // 2]

                # Update the frequency-domain plot
                self.lines[n_exg_channels + n].set_data(positive_frequencies, positive_magnitude)
                self.axes[n, 1].set_xlim(0, self.data.fs_exg / 2)  # Set xlim for frequency-domain plot
                # self.axes[n, 0].set_xlim((x[0], x[-1]))  # Set xlim for time-domain plot
                self.axes[n, 1].set_ylim(0, np.max(positive_magnitude))
                self.axes[n, 1].set_ylabel(f'Channel {n + 1}')
                if n == n_exg_channels - 1:
                    self.axes[n, 1].set_xlabel('Frequency [Hz]')
                if n == 0:
                    self.axes[n, 1].set_title('Frequency Domain')


        # Time of last update (must be inside axes for blit to work)
        duration = datetime.now() - self.init_time
        time_right = Viz_spec._format_time(duration)
        time_left = max(timedelta(seconds=0), duration - timedelta(seconds=self.window_secs))
        time_left = Viz_spec._format_time(time_left)
        time_txt_artists = []
        for ax in self.axes[-1, :]:
            time_txt_artists.append(ax.text(0, 0.05, time_left, transform=ax.transAxes, ha="left", size=9))
            time_txt_artists.append(ax.text(0, 0.05, time_right, transform=ax.transAxes, ha="right", size=9))

        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%H:%M:%S')
        [ax.xaxis.set_major_formatter(myFmt) for ax in self.axes[-1, :]]
        [ax.tick_params(
            axis="x", direction="in", pad=-15, size=9) for ax in self.axes[-1, :]]
        [ax.axes.xaxis.set_ticklabels([]) for ax in self.axes[-1, :]]


        # Gather artists (necessary if using FuncAnimation)
        lines_artists = self.lines
        xtick_artists = []
        for ax in self.axes[-1, :]:
            for artist in ax.get_xticklabels():
                artist.axes = ax
                xtick_artists.append(artist)
        # xtick_artists = [artist for ax in self.axes[-1, :] for artist in ax.get_xticklabels()]
        artists = (lines_artists + time_txt_artists + xtick_artists)

        return artists

    def close(self, _):
        self.data.is_connected = False
        print('Window closed.')

        # to start the animation

    # def start_animation(self, event):
    #     self.animation.event_source.start()
    #
    #     # to stop the animation
    #
    # def stop_animation(self, event):
    #     self.animation.event_source.stop()

    def start(self):

        from matplotlib.animation import FuncAnimation
        self.animation = FuncAnimation(self.figure, self.update,
                                  blit=True, interval=self.update_interval_ms, repeat=False, cache_frame_data=False)


        # # Create start and stop buttons
        # ax_start = plt.axes([0.05, 0.01, 0.05, 0.025])
        # button_start = Button(ax_start, 'Start')
        # button_start.on_clicked(self.start_animation)
        #
        # ax_stop = plt.axes([0.11, 0.01, 0.05, 0.025])
        # button_stop = Button(ax_stop, 'Stop')
        # button_stop.on_clicked(self.stop_animation)


        # plt.show()






# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.signal import get_window
#
# # Parameters
# fs = 500  # Sampling frequency
# window_size = int(0.5 * fs)  # Window size: 500 ms
# update_interval = int(0.1 * fs)  # Update interval: 100 ms
# duration = 10  # Duration of the signal in seconds
# t = 0  # initiate the time vector to construct the signal
# points_per_frame = 10  # Number of data points to add per frame
# num_channels = 4  # Number of channels
#
# # Initialize the plot
# fig, axs = plt.subplots(num_channels, 2, figsize=(12, 8), sharex='col')
#
# # Initialize time-domain lines and frequency-domain lines for each channel
# lines_time = []
# lines_freq = []
# for i in range(num_channels):
#     # Time-domain plot
#     line1, = axs[i, 0].plot([], [], lw=2)
#     axs[i, 0].set_ylim(-6, 6)
#     axs[i, 0].set_ylabel(f'Channel {i+1}\nAmplitude')
#
#     # Frequency-domain plot
#     line2, = axs[i, 1].plot([], [], lw=2)
#     axs[i, 1].set_xlim(0, fs/2)
#     axs[i, 1].set_ylim(0, 80)
#     axs[i, 1].set_ylabel('Magnitude')
#
#     lines_time.append(line1)
#     lines_freq.append(line2)
#
# # Initialize the data for real-time signals
# real_time_signals = [np.zeros(window_size) for _ in range(num_channels)]
# time_vector = np.linspace(0, window_size / fs, window_size)
#
# def init():
#     for line1, line2 in zip(lines_time, lines_freq):
#         line1.set_data([], [])
#         line2.set_data([], [])
#     return lines_time + lines_freq
#
# def update(frame):
#     global real_time_signals, time_vector, t
#
#     t += points_per_frame
#     vec_to_add = np.linspace(t, t + points_per_frame-1, points_per_frame) / fs
#     current_time = (t * points_per_frame) / fs
#     time_vector = np.linspace(current_time - window_size / fs, current_time, window_size)
#
#     for i in range(num_channels):
#         # Simulate real-time signal generation by adding multiple points per frame
#         new_points = (
#                 np.sin(2 * np.pi * 50 * vec_to_add)
#                 + np.sin(2 * np.pi * 120 * vec_to_add)
#                 + np.sin(np.random.normal(0, 20) * np.pi * vec_to_add)  # add random frequency of sine wave
#                 + np.random.normal(0, 0.5, points_per_frame)  # add random noise
#                       )
#         real_time_signals[i] = np.roll(real_time_signals[i], -points_per_frame)
#         real_time_signals[i][-points_per_frame:] = new_points
#
#         # Update the time vector for the time-domain plot
#         lines_time[i].set_data(time_vector, real_time_signals[i])
#         axs[i, 0].set_xlim(time_vector[0], time_vector[-1])
#
#         # Apply window function to the segment
#         window_function = get_window('hann', window_size)
#         windowed_segment = real_time_signals[i] * window_function
#
#         # Compute the FFT of the windowed segment
#         fft_result = np.fft.fft(windowed_segment)
#         fft_magnitude = np.abs(fft_result)
#         frequencies = np.fft.fftfreq(len(fft_magnitude), 1/fs)
#
#         # Only take the positive frequencies and corresponding magnitudes
#         positive_frequencies = frequencies[:len(frequencies)//2]
#         positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
#
#         # Update the frequency-domain plot
#         lines_freq[i].set_data(positive_frequencies, positive_magnitude)
#
#     return lines_time + lines_freq
#
# # Create animation
# ani = FuncAnimation(fig, update, frames=np.arange(0, duration*fs, update_interval // points_per_frame), init_func=init, blit=True, interval=100)
# plt.tight_layout()
# plt.show()