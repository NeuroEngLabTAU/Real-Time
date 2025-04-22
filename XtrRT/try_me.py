import pyxdf
from matplotlib import pyplot as plt

# Load the XDF file
path =r"C:\Users\YH006_new\Documents\delete_me\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
data, header = pyxdf.load_xdf(path)

# Display available streams
for i, stream in enumerate(data):
    stream_name = stream['info']['name'][0]
    stream_type = stream['info']['type'][0] if 'type' in stream['info'] else "Unknown"
    print(f"Stream Index: {i}, Name: {stream_name}, Type: {stream_type}")



stream_name = data[0]['info']['name'][0]
time_stamps =  data[0]['time_stamps']
time_stamps_sec = [time - time_stamps[0] for time in time_stamps]

fs_hz = float(data[0]['info']['effective_srate']) #maybe nomina;ls_rate


fig, axs = plt.subplots(3, 1, figsize=(10, 8))
for i in range(3):
    ax = axs[i]
    ax.set_title(f"Channel {i+1}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    signal = data[0]['time_series'][:, i]
    signal = data[0]['time_series'][:,i]
    ax.plot(time_stamps_sec, signal)

fig.savefig('input_signal.png', dpi=300, bbox_inches='tight')

fig.show()