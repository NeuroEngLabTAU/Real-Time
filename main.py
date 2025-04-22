import matplotlib
import pandas as pd
from pylsl.pylsl import resolve_stream, StreamInlet

from XtrRT.facial_image import Y_COOR, X_COOR, image_load, IMAGE_PATH

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from XtrRT.data import Data  # data collection
from XtrRT.electrodes_raw_streaming import Electrodes_Raw_Streaming  # stream raw data with electrodes location

# Define the desired visualisation
VIZ_RAW = True  # Raw signal with facial electrodes
SAMPLING_RATE_Hz = 500  # Sampling rate in Hz
VIZ_FILTERS = {'bandpass': [30, SAMPLING_RATE_Hz / 2 - 1], 'comb': [50, 100]}

# Define LSL parameters
IS_LSL_INLET = False  # LSL inlet for trigger stream
LSL_INLET_NAME = "XtrRT Inlet"  # Name of the LSL inlet stream

if __name__ == '__main__':
    lsl_inlet = None
    if IS_LSL_INLET:
        # Initialize LSL inlet for "MyTriggerStream"
        try:
            # Adjust this to match the stream from 'experiment_waldo_intensity.py'
            streams = resolve_stream('name', LSL_INLET_NAME)  # Resolving by stream name
            lsl_inlet = StreamInlet(streams[0])
            print(f"LSL inlet '{LSL_INLET_NAME}' detected.")
        except Exception as e:
            print("Error initializing LSL inlet:", e)

    image, height, width = image_load(IMAGE_PATH)

    # Prepare the data object
    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf",
                lsl_inlet=lsl_inlet)
    data.start()

    data.add_annotation("Start recording")

    if VIZ_RAW:
        electrodes_fig, electrodes_axes = plt.subplots()
        elctrodes_streaming = Electrodes_Raw_Streaming(data, window_secs=2.5, plot_exg=True, plot_imu=False,
                                                       filters=VIZ_FILTERS, update_interval_ms=100,
                                                       ylim_exg=(-500, 500), max_points=None, max_timeout=15,
                                                       x_coor=X_COOR, y_coor=Y_COOR, width=width, height=height,
                                                       image=image, filter_data=True,
                                                       figure=electrodes_fig, axes=electrodes_axes)

        electrodes_viz = elctrodes_streaming.start()

    plt.show()

    data.add_annotation("Stop recording")
    data.stop()

    data.join()  # Wait for the data thread to finish
    print('Main code terminated')

    print(data.annotations)
    print('Process terminated')

    # Export the annotations to a CSV file
    df_annotations = pd.DataFrame(data.annotations, columns=['Onset', 'Duration', 'Description'])
    df_annotations.to_csv(f'{data.save_as}_annotations.csv', index=False)
    print('Annotations exported to CSV')
