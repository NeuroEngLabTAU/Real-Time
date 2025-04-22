# Build a script that transmits data to a LSL stream
import time

from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np

# Define the stream parameters
STREAM_NAME = "XtrRT Inlet"
STREAM_TYPE = "Triggers"
STREAM_CHANNELS = 1
STREAM_SAMPLING_RATE = 0  # 0 for non-timed streams
STREAM_CHANNEL_FORMAT = 'string'  # Format of the data being streamed

lsl_info = StreamInfo(STREAM_NAME, STREAM_TYPE, STREAM_CHANNELS, STREAM_SAMPLING_RATE, STREAM_CHANNEL_FORMAT, "myuid34234")
lsl_outlet = StreamOutlet(lsl_info)
print(f"LSL stream '{STREAM_NAME}' is now running...")

while True:
    current_time = local_clock()
    trigger_value = f"Trigger at {current_time}"
    lsl_outlet.push_sample([trigger_value])
    print(f"Pushed sample: {trigger_value}")
    time.sleep(30)