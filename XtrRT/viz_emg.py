
from XtrUtils.utils import Utils
from XtrViz.plotter import Plotter
from XtrEMG.detection import detect_emg
from XtrUtils.filterbank import Filterer


def plot_emg(emg_axes, x, y, fs):
    patch_artists = []
    # [patch.remove() for patch in self._existing_patches]
    for n in range(len(emg_axes)):
        filters = {'highpass': {'W': 30}, 'comb': {'W': 50}}
        filt = Filterer.filter_data(y[:, n], filters, fs, verbose=False)
        is_emg = detect_emg(filt, fs=fs, emg_thresh=5e6, fft_validate=True, line_interference_validate=True,
                            verbose=False)
        ons, offs = Utils.get_onsets_offsets(is_emg)
        patches = Plotter.add_patches(x, ons, offs, emg_axes[n, 0], label="EMG")
        patch_artists.extend(patches)

    return patch_artists