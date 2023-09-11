
import numpy as np
from time import time
from datetime import datetime as dt
from pylsl import resolve_streams, StreamInlet, StreamInfo, StreamOutlet

from XtrRT.data import Data, ConnectionTimeoutError


class DataLSL(Data):

    def __init__(self, verbose: bool = False, save_as: str = None, timeout: float = 10, stream_name: str = "Real-time EMG data" , modality: str = 'EMG'):

        Data.__init__(self, host_name="127.0.0.1", port=20001, timeout_secs=timeout, verbose=verbose, save_as=save_as)

        is_streaming = False
        now = time()
        while not is_streaming:

            # Quit if fail to receive data
            if time() - now >= timeout:
                raise ConnectionTimeoutError

            rec = self._parse_incoming_records()
            if any(rec):
                self.fs = rec[0].fs
                self.n_channels = rec[0].data.shape[1]
                is_streaming = True

        self.stream_info = StreamInfo(name=stream_name,
                                      type=modality,
                                      channel_count=self.n_channels,
                                      nominal_srate=self.fs,
                                      channel_format="float32",
                                      source_id='',
                                      )

    @staticmethod
    def get_chunk(records):

        # TODO: check for missing data or no???
        chunk = np.vstack([record.data for record in records if record.record_type == "EXG"])
        chunk = [list(row) for row in chunk]

        return chunk

    def run(self):

        outlet = StreamOutlet(self.stream_info)
        while self.is_connected:

            # Receive incoming record(s)
            records = self._parse_incoming_records()

            # Convert records list into single 2D numpy array (data `chunk`) for LSL outlet input
            chunk = DataLSL.get_chunk(records)

            outlet.push_chunk(chunk, timestamp=0.0, pushthrough=True)

        print('Connection terminated.')


def stream_inlet(timeout: float, verbose: bool = True, stream_name: str = "Real-time EMG data" , modality: str = 'EMG'):

    # Find available LSL data streams
    print(f"Searching for data streams...")
    streams = resolve_streams(wait_time=3)
    # streams = resolve_byprop('type', data_source, timeout=timeout)
    if not any(streams):
        print(f"Failed to find any data streams")
        exit(1)

    # Find streams connected to this computer's LSL either locally or via network specified in lsl_api.cfg
    inlets = []
    data_types = []

    for stream in streams:
        print(f"Found stream: Name={stream.name()}, Type={stream.type()}")
        if stream.name() == stream_name and stream.type() == modality:
            inlets = StreamInlet(stream)
            break

    for stream in streams:
        inlet = StreamInlet(stream, max_chunklen=0)
        inlets.append(inlet)
        info = inlet.info()
        data_type = info.type()
        data_types.append(data_type)
    print(f"Streaming {data_types} data...")

    # # Initiate data streamer
    # # print(f"Streaming {data_source} data.")
    # inlet = StreamInlet(streams[0], max_chunklen=0)
    # info = inlet.info()
    # description = info.desc()

    # # Get channel info
    # n_channels = info.channel_count()
    # ch = description.child('channels').first_child()
    # ch_names = [ch.child_value('label')]
    # for i in range(1, n_channels):
    #     ch = ch.next_sibling()
    #     ch_names.append(ch.child_value('label'))

    # Stream data
    data = []
    timestamps = []
    t_init = time()
    while True:

        # Stop streaming after `timeout` seconds
        if time() - t_init > timeout:
            break

        for inlet in inlets:
            try:

                # Accept new data from stream
                samples, timestamp = inlet.pull_sample()  # inlet.pull_chunk()

                # If new data exists, add it to data (which, in its current form, is a list of (nchannels,) samples)
                if timestamp:
                    data.append(samples)
                    timestamps.append(timestamp)  # timestamps.extend(timestamp) if pull_chunk

                    # Print stream name and received samples to console
                    if verbose:
                        print(f'{dt.now().strftime("%H:%M:%S")}\t {len(samples)} {inlet.info().type()} samples')

            except KeyboardInterrupt:
                break

    # Print LSL "time correction" for each received stream
    for inlet in inlets:
        time_correction = inlet.time_correction()
        print(f"{inlet.info().type()} time correction: {time_correction}")

    # TODO: save data


if __name__ == '__main__':

    # This collects data from a DAU connected to this computer and sends it to LSL
    outlet_stream = DataLSL(stream_name="Real-time EMG data" , modality='EMG')
    outlet_stream.start()

    # This receives all LSL streams that are connected to the network defined in lsl_api.cfg (i.e. via `KnownPeers`)
    timeout = 15   # Stop after `timeout` seconds
    stream_inlet(timeout=timeout, verbose=True, stream_name="Real-time EMG data" , modality='EMG')

    print("Done recording.")

