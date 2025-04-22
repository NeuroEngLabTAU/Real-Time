import pyedflib

# Replace with your EDF file path
edf_file = 'test.edf'

with pyedflib.EdfReader(edf_file) as f:
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    annotations = f.readAnnotations()

    print("Signal Labels:", signal_labels)
    print("Annotations:", annotations)