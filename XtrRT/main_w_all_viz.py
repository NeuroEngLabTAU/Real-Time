import matplotlib

from XtrRT.facial_image import get_dummy_heatmap

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from XtrRT.data import Data  # data collection
from XtrRT.viz import Viz  # real time raw data plotting
# from XtrRT.ica import Viz_ICA   # interactive ica in real time
from XtrRT.ica_streaming import Viz_ICA_Streaming  # Streaming ICA
from XtrRT.electrodes_raw_streaming import Electrodes_Raw_Streaming  # stream raw data with electrodes location
from XtrRT.Spectogram import Viz_spec  # spectogram

# Define the desired visualisation
viz_raw = False  # Aaron's real time raw data plotting
viz_spectogram = False  # spectogram streaming
viz_ica = False  # Bara's semi real-time ICA
viz_ica_streaming = False  # streaming ICA signals with heatmaps
Electrodes_raw = True  # raw signal electrodes
IS_LSL_INLET = True  # LSL inlet for trigger stream
LSL_INLET_NAME = "XtrRT Inlet"  # Name of the LSL inlet stream

if __name__ == '__main__':
    lsl_inlet = None
    if IS_LSL_INLET:
        # Initialize LSL inlet for "MyTriggerStream"
        try:
            # Adjust this to match the stream from 'experiment_waldo_intensity.py'
            streams = resolve_stream('name', LSL_INLET_NAME)  # Resolving by stream name
            lsl_inlet = StreamInlet(streams[0])
        except Exception as e:
            print("Error initializing LSL inlet:", e)

    d_interpolate = get_dummy_heatmap()
    image, height, width = image_load(IMAGE_PATH)


    # Prepare the data object
    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf", lsl_inlet=lsl_inlet)
    data.start()

    data.add_annotation("Start recording")

    filters = {'highpass': {'W': 30}, 'comb': {'W': [50, 100]}}

    if viz_raw:
        raw_fig = plt.figure()
        raw_streaming = Viz(data, window_secs=10, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False,
                            filters=filters,
                            update_interval_ms=10, ylim_exg=(-250, 250), max_points=None, max_timeout=15,
                            filter_data=True, fig=raw_fig)
        raw_viz = raw_streaming.start()

    if viz_spectogram:
        # spec_fig = plt.figure()
        spec_streaming = Viz_spec(data, window_secs=10, plot_exg=True, plot_spectogram=True, find_emg=False,
                                  filters=filters,
                                  update_interval_ms=10, ylim_exg=(-350, 350), max_points=None, max_timeout=15,
                                  filter_data=False)
        spec_viz = spec_streaming.start()
    # if viz_ica:
    #     viz = Viz_ICA(data, window_secs=10, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
    #               update_interval_ms=10, ylim_exg=(-250, 250), max_points=None, max_timeout=15,
    #               x_coor=x_coor, y_coor=y_coor, width=width, height=height, image=image, filter_data=True)

    if viz_ica_streaming:
        ica_fig = plt.figure()
        viz = Viz_ICA_Streaming(data, lsl_inlet=lsl_inlet, window_secs=10, plot_exg=True, plot_imu=False,
                                plot_ica=False, find_emg=False, filters=filters,
                                update_interval_ms=10, ylim_exg=(-5, 5), max_points=None, max_timeout=15,
                                x_coor=x_coor, y_coor=y_coor, width=width, height=height, image=image,
                                d_interpolate=d_interpolate, filter_data=True)
        # figure=ica_fig
        ica_viz = viz.start()

    if Electrodes_raw:
        electrodes_fig, electrodes_axes = plt.subplots()
        elctrodes_streaming = Electrodes_Raw_Streaming(data, window_secs=2.5, plot_exg=True, plot_imu=False,
                                                       plot_ica=False,
                                                       find_emg=False, filters=filters, update_interval_ms=100,
                                                       ylim_exg=(-250, 250), max_points=None, max_timeout=15,
                                                       x_coor=x_coor, y_coor=y_coor, width=width, height=height,
                                                       image=image, filter_data=True,
                                                       figure=electrodes_fig, axes=electrodes_axes)

        electrodes_viz = elctrodes_streaming.start()

    plt.show()

    data.add_annotation("Stop recording")
    data.stop()

    data.join()  # Wait for the data thread to finish
    print('Main code terminated')

    print(data.annotations)
    print('process terminated')
