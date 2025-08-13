# Real-Time
Built on [gittXtrodes' real-time](https://github.com/xtrodesorg/XtrRT). Make sure to clone their repo (instructions in
our [Drive](https://drive.google.com/drive/folders/1vUUEzNRzD2WzMDtzkNSd3YFly1mIQcKq?usp=sharing)).

Includes main that imports the Data class to collect data via Bluetooth and three visualization tools:
1. Viz: streaming of real-time raw data.
2. Viz_ICA_Streaming: streaming of ICA signals + heatmaps of real-time data.
3. VIZ_ICA: ICA interactive analysis of real-time data.

## Known Issues
- Annotation bug: not all annotations are being saved, when many annotations are sent in a short time
    - Temp solution is to always save annotations in a separate csv file.
- When recording the IMU data, a delay develops and after ~21min there is huge delay between IMU and EXG, and the EXG begins to lag.
    - Temp solution is to skip the IMU data records.
    - Until 20min of recording, the EXG and IMU data looks fine.
    - TO DO: 
        - Investigate the delay and fix it.
        - Add it back to the recordings (see `records.py` for how to add it back).
- Data is not cleaned, saving the raw data is done after finishing the recording, which is not ideal and challenging for long-term recordings.
  - Or it can be cuz of plotting the whole data each time, need to check.
  - Testing 30min recording was fine, after that there was lag in the data.
  - TO DO: 
      - Consider flushing the data to csv, and maybe generating the edf at the end using the csv.
