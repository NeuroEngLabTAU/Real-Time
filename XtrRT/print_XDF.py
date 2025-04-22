import pyxdf
import matplotlib.pyplot as plt

# Path to your XDF file
xdf_path = fr"C:\Users\YH006_new\Documents\CurrentStudy\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

# Load the XDF file
data, header = pyxdf.load_xdf(xdf_path)

for stream in data:
    print(f"Stream: {stream['info']['name'][0]}")
    print(f"Number of Channels: {stream['info']['channel_count']}")
    print(f"Sampling Rate: {stream['info']['nominal_srate'][0]}")
    print(f"Format: {stream['info']['channel_format'][0]}")
    try:
        print(f"Data shape: {stream['time_series'].shape}")
    except AttributeError:
        print("No time series data available.")
    # Access the time series and timestamps
    time_series = stream['time_series']
    timestamps = stream['time_stamps']
    # Do something with the time_series and timestamps...

    if len(time_series) > 1 and stream['info']['channel_format'][0] != 'string':
        # Example: Plotting the data of the first stream
        plt.plot(stream['time_stamps'], stream['time_series'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Data from Stream: ' + stream['info']['name'][0])
        plt.show()
    else:
        print(time_series)