
from XtrRT.data import Data
#from XtrRT.viz import Viz
#from XtrRT.vizTestAnimator import Viz
#from XtrRT.transorbitStim_Blinks import Viz
from XtrRT.transorbitStim import Viz


if __name__ == '__main__':

    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="testBlink_noData2.edf")
    data.start()

    data.add_annotation("Start recording")
    filters = {'highpass': {'W': 30}, 'comb': {'W': 50}}

# ----------------------  VISUAL EXPERIMENT CODE ----------------------
   # data.data.exg_data[-120:, :]


# ------------------------------------------------------------------------------------------


            # Shira : window sec is 3 so that there will be a single blink in each window !

    #viz = Viz(data, window_secs=3, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
    #          update_interval_ms=10, ylim_exg=(-500, 500), max_points=None, max_timeout=15)
    #viz.start()

    #vizTestAnimator = Viz(data, window_secs=5, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
    #          update_interval_ms=10, ylim_exg=(-500, 500), max_points=None, max_timeout=15)
    #vizTestAnimator.start()

    #transorbitStim_Blinks = Viz(data, window_secs=5, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
    #          update_interval_ms=500, ylim_exg=(-500, 500), max_points=None, max_timeout=15)
    #transorbitStim_Blinks.start()

    transorbitStim = Viz(data, window_secs=10, plot_exg=True, plot_imu=False, plot_ica=False, find_emg=False, filters=filters,
              update_interval_ms=500, ylim_exg=(-500, 500), max_points=None, max_timeout=15)
    #transorbitStim.newvisual()
    transorbitStim.start()



    #transorbitStim.visual_experiment()
    # the code does not update the RT ! so you get only 5 sec!!

    #transorbitStim.saccadeDetection()

    data.add_annotation("Stop recording")
    data.stop()

    print(data.annotations)
    print('process terminated')

# shira :
# make use that the electrodes are always placed 2 traces down! otherwise the indexing will be wrong.
#