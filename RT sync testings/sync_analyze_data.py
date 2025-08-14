import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import find_peaks

mpl.use('Qt5Agg')  # Use Qt5 backend for matplotlib


def load_edf(file_path):
    return pyedflib.EdfReader(file_path)


def get_triggers_annotations(edf_file):
    annotations = edf_file.readAnnotations()
    annotations_df = pd.DataFrame({
        'onset': annotations[0],
        'duration': annotations[1],
        'description': annotations[2]
    })
    trigger_annotations = annotations_df[
        annotations_df['description'].str.contains('trigger', case=False, na=False)
    ]
    return trigger_annotations['onset'].values


def get_top_n_peaks(data, min_distance, n):
    # Find all peaks
    peaks, properties = find_peaks(data, distance=min_distance)

    # Get peak heights
    peak_heights = data[peaks]

    # Sort peaks by height (descending) and get top n
    if n > len(peaks):
        n = len(peaks)
    sorted_indices = np.argsort(peak_heights)[::-1][:n]
    top_n_peaks = peaks[sorted_indices]
    top_n_heights = peak_heights[sorted_indices]

    return top_n_peaks, top_n_heights


def detect_sin_onsets(data, fs,
                      threshold=None,
                      threshold_factor=0.2,
                      min_distance_sec=0.8):
    # Calculate signal power using sliding window
    window_size = int(0.1 * fs)  # 100ms window
    power = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='same')

    # Smooth the power signal
    from scipy.signal import savgol_filter
    power_smooth = savgol_filter(power, window_length=3, polyorder=2)

    # Set detection threshold based on first largest 5 peaks and baseline statistics
    if threshold is None:
        avg_peaks = get_top_n_peaks(power_smooth, min_distance=int(1 * fs), n=5)[1].mean()
        threshold = threshold_factor * avg_peaks

    # Find onset points where power crosses threshold upward
    min_distance = int(min_distance_sec * fs)

    # Detect threshold crossings
    above_threshold = power_smooth > threshold
    crossings = np.diff(above_threshold.astype(int))
    onset_indices = np.where(crossings == 1)[0]  # Rising edges only

    # Filter out detections too close together
    if len(onset_indices) > 1:
        filtered_onsets = [onset_indices[0]]
        for onset in onset_indices[1:]:
            if onset - filtered_onsets[-1] >= min_distance:
                filtered_onsets.append(onset)
        onset_indices = np.array(filtered_onsets)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.plot(np.arange(len(data)) / fs, data, label='Raw Signal', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('Raw Signal Amplitude')
    ax.set_xlabel('Time (s)')

    ax2 = ax.twinx()  # Create a second y-axis for the power signal
    ax2.plot(np.arange(len(power_smooth)) / fs, power_smooth, label='Power Signal', color='red', alpha=0.7)
    ax2.set_ylabel('Power Signal')
    ax2.axhline(threshold, color='orange', linestyle='--', label=f'Threshold ({threshold:.3f})', alpha=0.7)
    for onset in onset_indices:
        ax2.axvline(onset / fs, color='green', linestyle='--', alpha=0.7,
                    label='Detected Onset' if onset == onset_indices[0] else "")

    axes[1].plot(onset_indices[:-1] / fs, np.diff(onset_indices) / fs, marker='o', linestyle='-', color='purple')

    return onset_indices / fs, power_smooth, threshold, data


def calculate_delays(trigger_annot_times, onset_times):
    delays = []
    matched_triggers_annot = []
    matched_onsets = []

    for trigger in trigger_times:
        future_onsets = onset_times[onset_times > trigger]
        if len(future_onsets) > 0:
            next_onset = future_onsets[0]
            delays.append(next_onset - trigger)
            matched_triggers.append(trigger)
            matched_onsets.append(next_onset)

    return np.array(delays), np.array(matched_triggers_annot), np.array(matched_onsets)


def plot_analysis(times, data, onset_times, trigger_times, delays, matched_triggers,
                  power_signal, threshold, raw_data, channel_name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Original signal with annotations
    axes[0].plot(times, data, 'b-', alpha=0.6, linewidth=0.8)
    for i, t in enumerate(trigger_times):
        axes[0].axvline(t, color='red', linestyle='--', alpha=0.7,
                        label='Triggers' if i == 0 else "")
    for i, t in enumerate(onset_times):
        axes[0].axvline(t, color='green', alpha=0.7,
                        label='Detected Onsets' if i == 0 else "")
    axes[0].set_title(f'{channel_name}: Raw Signal with Triggers and Onsets')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Power signal and threshold
    axes[1].plot(times, power_signal, 'r-', linewidth=1.5, label='Signal Power')
    axes[1].axhline(threshold, color='orange', linestyle='--',
                    label=f'Threshold ({threshold:.3f})')
    for t in onset_times:
        axes[1].axvline(t, color='green', alpha=0.5)
    axes[1].set_title('Signal Processing: Power Signal and Detection Threshold')
    axes[1].set_ylabel('Power')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Delay over time
    if len(delays) > 0:
        delays_ms = delays * 1000
        axes[2].plot(matched_triggers, delays_ms, 'o-', color='purple',
                     markersize=5, linewidth=1.5)
        axes[2].set_title('Delay Variation Over Time')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Delay (ms)')
        axes[2].grid(True, alpha=0.3)

        # Add mean line
        axes[2].axhline(np.mean(delays_ms), color='red', linestyle='--', alpha=0.7,
                        label=f'Mean: {np.mean(delays_ms):.1f} ms')
        axes[2].legend()

    plt.tight_layout()
    plt.show()

    return fig


def analyze_stimulation_delays(file_path, channel_index, threshold_factor=3):
    # Load data
    f = load_edf(file_path)
    channel_data = f.readSignal(channel_index)
    fs = f.getSampleFrequency(channel_index)
    channel_label = f.getLabel(channel_index)
    times = np.arange(len(channel_data)) / fs

    print(f"File Analysis:")
    print(f"  Channel: {channel_label} (index {channel_index})")
    print(f"  Sampling Rate: {fs} Hz")
    print(f"  Duration: {len(channel_data) / fs:.1f} seconds")
    print()

    # Extract triggers and detect onsets
    trigger_annot_times = get_triggers_annotations(f)
    onset_times, envelope, threshold, filtered_data = detect_sin_onsets(data=channel_data, fs=fs,
                                                                        min_distance_sec=1.1,  # pulses of 1 sec
                                                                        )

    print(f"Detection Results:")
    print(f"  Triggers found: {len(trigger_annot_times)}")
    print(f"  Onsets detected: {len(onset_times)}")

    onset_intervals = np.diff(onset_times)
    trigger_intervals = np.diff(trigger_annot_times)

    fig_intervals, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(trigger_annot_times[:-1], onset_intervals, marker='o', linestyle='-', color='purple', label = 'Onset Intervals')
    axes[0].plot(trigger_annot_times[:-1], trigger_intervals, marker='o', linestyle='-', color='orange', label='Trigger Intervals')
    axes[0].set_ylabel('Interval (s)')
    axes[0].legend()

    axes[1].plot(trigger_annot_times, onset_times - trigger_annot_times, marker='o', linestyle='-', color='green',
                 label='Onset - Trigger')
    axes[1].set_xlabel('Trigger (s)')
    axes[1].set_ylabel('Onset - Trigger (s)')
    # axes[1].plot(trigger_annot_times, onset_times, marker='o', linestyle='-', color='green',
    #              label='Onset Times')
    # axes[1].plot(trigger_annot_times, trigger_annot_times,
    #              marker='x', linestyle='-', color='blue', alpha=0.5, label='Trigger Times ')
    # axes[1].set_ylabel('(s)')
    # axes[1].legend()

    # Calculate delays
    matched_triggers_annot = trigger_annot_times
    matched_onsets = onset_times
    delays = onset_times - trigger_annot_times

    print(f"  Matched pairs: {len(delays)}")
    print()

    # Statistics
    if len(delays) > 0:
        delays_ms = delays * 1000
        print(f"Mean intervals [sec]: Onset {np.mean(onset_intervals):.2f} vs Trigger {np.mean(trigger_intervals):.2f}")
        print(f"Difference between the intervals: {np.mean(onset_intervals - trigger_intervals):.2f} seconds")
        print(f'Expected delay after 30 pulse/min of stimulation: {np.mean(onset_intervals - trigger_intervals) * 30:.2f} seconds')
        print(f'--------------------')
        print(f"Delay Statistics:")
        print(f"  Mean: {np.mean(delays_ms):.2f} ± {np.std(delays_ms):.2f} ms")
        print(f"  Median: {np.median(delays_ms):.2f} ms")
        print(f"  Range: {np.min(delays_ms):.2f} - {np.max(delays_ms):.2f} ms")
        print()

        # Create results dataframe
        results_df = pd.DataFrame({
            'trigger_time_s': matched_triggers_annot,
            'onset_time_s': matched_onsets,
            'delay_ms': delays_ms
        })

        # Generate plots
        fig = plot_analysis(times, channel_data, onset_times, trigger_annot_times,
                            delays, matched_triggers_annot, envelope, threshold,
                            filtered_data, channel_label)

        f.close()
        return results_df, fig

    else:
        print("No matching trigger-onset pairs found!")
        print("Try adjusting threshold_factor or check your data.")
        f.close()
        return None, None


# Main execution
if __name__ == "__main__":

    # Configuration
    file_path = r"D:\Users\Rawan\RT sync testings\data\2025-08-13 EXG test with STG4\session1_31min_stim_sin_40Hz_10mV.edf"
    channel_index = 2  # Channel 3 (0-indexed)

    # Run analysis
    results_df, fig = analyze_stimulation_delays(
        file_path=file_path,
        channel_index=channel_index,
        threshold_factor=3
    )

    plt.show()

    # Save results
    if results_df is not None:
        output_file = "results/stimulation_delay_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        # Display sample results
        print(f"\nSample Results:")
        print(results_df.head(10).to_string(index=False))
