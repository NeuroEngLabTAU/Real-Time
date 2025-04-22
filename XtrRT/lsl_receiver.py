from pylsl import resolve_streams, StreamInlet

def receive_triggers():
    # Resolve the streams
    streams = resolve_streams()

    # Look for the desired stream, change 'TriggerStream' to your stream name
    target_stream = next((s for s in streams if s.name() == "XtrRT Inlet"), None)

    if target_stream:
        # Create a stream inlet
        inlet = StreamInlet(target_stream)

        # Receive triggers
        while True:
            sample, timestamp = inlet.pull_sample()
            if sample:
                # Sample contains the trigger string
                print(f"Received trigger: {sample[0]} at timestamp {timestamp}")
    else:
        print("Desired stream not found")

if __name__ == '__main__':
    receive_triggers()