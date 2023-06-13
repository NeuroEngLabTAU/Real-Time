##############################
###### Import libraries ######
##############################
import pyedflib  # to convert EDF file to a numpy array, install 'pyEDFlib'
import cv2  #for image loading+user electrode location selection, install 'opencv-python'
#for ICA + time plots + heatmap + spectral analysis
from scipy.stats import kurtosis
from picard import picard #*need to cite*,  install "python-picard"
from numpy.linalg import inv
from scipy.interpolate import griddata
import matplotlib.image as mpimg
from matplotlib.pyplot import cm #colormaps
import matplotlib.ticker as ticker
from scipy import signal
import math
from matplotlib.widgets import Slider #for horizontal scrolling in time plot
#generic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tkinter as tk
from matplotlib.widgets import CheckButtons
from datetime import datetime


sampling_rate=250

wanted_order = np.array([12, 13, 14, 15,
                             11, 10, 9, 8,
                             4, 5, 6, 7,
                             3, 2, 1, 0])


times={}
start_relax=datetime.now()
end_expression=datetime.now()


colormap=cm.get_cmap("tab20c")

#########################
###### Load image #######
#########################
def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width

def red_size(arr, thresh):
    myrange = np.nanmax(arr) - np.nanmin(arr)
    norm_arr = (arr - np.nanmin(arr)) / myrange
    # threshold for red/orange 0.75-0.1
    condition = (norm_arr.ravel()) > thresh
    flat_indices = np.where(condition)[0]

    return flat_indices, flat_indices.shape[0]

def norm(arr):
    myrange = np.nanmax(arr) - np.nanmin(arr)
    norm_arr = (arr - np.nanmin(arr)) / myrange
    return norm_arr


def atlas_duplicates_sig(duplicates, potential_electrodes, order_electrode, f_interpolate, flat_coordinates, sig):
    for duplicate in duplicates:
        flat_indices = []
        flat_indices_sizes = []
        condition = lambda x: x == duplicate
        indices = [index for index, element in enumerate(order_electrode) if index in sig and condition(element)]
        for index in indices:
            thresh = 0.75
            array, size = red_size(f_interpolate[index], thresh)
            flat_indices.append(array)
            flat_indices_sizes.append(size)
        # change the electrode number assignment for those that aren't min
        not_noise = np.argmin(flat_indices_sizes)
        for i in range(len(flat_indices)):  # thinking of looping not sequtially but by less noisy to most noisy
            if (i != not_noise):
                index = indices[i]
                array = flat_indices[i]
                noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                    0]]  # find the electrodes included in the red area
                # common_electrodes=[el for el in common_electrodes if el not in duplicates] #exlude duplicates from list
                while (len(common_electrodes) == 0):
                    thresh = thresh - 0.05  # might make harsher - 0.75 - 0.5 - 0.25 steps
                    array, size = red_size(f_interpolate[index], thresh)
                    noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                    common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                        0]]  # find the electrodes included in the red area
                red_values = np.zeros(len(common_electrodes))
                for j in range(len(common_electrodes)):
                    red_values[j] = f_interpolate[index][
                        y_coor[common_electrodes[j]], x_coor[common_electrodes[j]]]  # find the 'redness' value

                chosen_electrode = common_electrodes[np.argmax(red_values)]  # assign the electrode that is most red
                order_electrode[indices[i]] = chosen_electrode

                # update potential electrode list
                potential_electrodes = potential_electrodes[potential_electrodes != chosen_electrode]

    return potential_electrodes, order_electrode


def atlas_duplicates_not_sig(duplicates, potential_electrodes, order_electrode, f_interpolate, flat_coordinates, sig):
    for duplicate in duplicates:
        flat_indices = []
        flat_indices_sizes = []
        condition = lambda x: x == duplicate
        indices = [index for index, element in enumerate(order_electrode) if condition(element)]
        for index in indices:
            thresh = 0.75
            array, size = red_size(f_interpolate[index], thresh)
            flat_indices.append(array)
            flat_indices_sizes.append(size)

        common_values = np.intersect1d(indices, sig)

        if common_values.size == 0:  # if sig is not included in index, chose min area as the non_noise channel
            not_noise = np.argmin(flat_indices_sizes)
        else:  # otherwise, the significant channel is the not noise channel
            not_noise = np.where(np.isin(indices, common_values))[0][0]

        for i in range(len(flat_indices)):  # thinking of looping not sequtially but by less noisy to most noisy
            thresh = 0.75
            if (i != not_noise):
                index = indices[i]
                array = flat_indices[i]
                noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                    0]]  # find the electrodes included in the red area
                while (len(common_electrodes) == 0):
                    thresh = thresh - 0.05  # might make harsher - 0.75 - 0.5 - 0.25 steps
                    if (thresh <= 0.5):
                        chosen_electrode = -1  # assign -1: noisy channel that would be put aside
                        break
                    array, size = red_size(f_interpolate[index], thresh)
                    noise_electrodes = np.nonzero(np.in1d(flat_coordinates, array))[0]
                    common_electrodes = noise_electrodes[np.nonzero(np.in1d(noise_electrodes, potential_electrodes))[
                        0]]  # find the electrodes included in the red area
                if (thresh > 0.5):
                    red_values = np.zeros(len(common_electrodes))
                    for j in range(len(common_electrodes)):
                        red_values[j] = f_interpolate[index][
                            y_coor[common_electrodes[j]], x_coor[common_electrodes[j]]]  # find the 'redness' value

                    chosen_electrode = common_electrodes[np.argmax(red_values)]  # assign the electrode that is most red
                order_electrode[indices[i]] = chosen_electrode

                # update potential electrode list
                potential_electrodes = potential_electrodes[potential_electrodes != chosen_electrode]

    return potential_electrodes, order_electrode


#picture paramters
picture_type=45
image_path = r"C:\Users\YH006_new\Simulation\Paul_%d.jpg"%picture_type
image, height, width = image_load(image_path)
sigbufs=np.load(r'C:\Users\YH006_new\Simulation\sigbufs.npy')
x_coor=np.load(r'C:\Users\YH006_new\Simulation\x_coor_%d.npy'%picture_type)
y_coor=np.load(r'C:\Users\YH006_new\Simulation\y_coor_%d.npy'%picture_type)
grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
points = np.column_stack((x_coor, y_coor))

#just for now, for the simulation
#paths
annotations_path=r"C:\Users\YH006_new\Simulation\MAtlas.xlsx"
#load annotations (excel)
annotations = pd.read_excel(annotations_path)
relevant_annotations=annotations["Annotation"].to_numpy()
relevant_times =annotations["Time (total)"].to_numpy()


def obtain_sources_for_segement(i, ica_time=None):
    #i is the input value

    ann = relevant_annotations[i]
    ob=int(offset_before * sampling_rate)
    oa=int(offset_after * sampling_rate)

    #offset = 2
    start_time = relevant_times[i] * sampling_rate - ob
    end_time = relevant_times[i + 1] * sampling_rate + oa

    #if ica_time isn't specified then calculate over the expression+offsets
    if(not ica_time):
        data = sigbufs[:, start_time: end_time]
    #otherwise calculate the entire said time up to end time
    else:
        data = sigbufs[:, end_time-ica_time: end_time]

    # input data (X) needs to be #of signals by time
    K, W, Y = picard(data, n_components=16, ortho=True,
                     max_iter=200)

    duration=end_time-start_time
    print('ica_time', ica_time)
    relax_calib_per_expression = Y[:, -duration:-duration+ob]
    seg_sources = Y[:, -duration+ob:-oa]
    relative_intensity = np.average(np.abs(seg_sources), axis=1) / np.average(np.abs(relax_calib_per_expression),
                                                                              axis=1)
    sort_intensity = np.argsort(relative_intensity)


    return Y, K, W, sort_intensity, relative_intensity, start_time, end_time

def calculate_heatmaps(relative_intensity, sort_intensity, W,K):
    #calculates heatmaps and finds inital guesses for the electrodes

    # threshold for significance 1.8 (after rounding)
    number_of_sig = len(np.where(np.round(relative_intensity[sort_intensity], 2) >= 1.8)[0])
    sig = sort_intensity[-number_of_sig:]

    inverse = np.absolute(inv(np.matmul(W, K)))
    f_interpolate = []
    # plot the 16 electrodes in a way that makes sense
    order_electrode = np.full((16,), -1)

    for i in range(16):  #
        # f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))
        interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')

        norm_arr = norm(interpolate_data)
        f_interpolate.append(norm_arr)

        # if i in sig:
        ## find red point
        electrode = np.argmax(inverse[:, i])
        order_electrode[i] = electrode

    return f_interpolate, order_electrode, sig


def calculate_atlas(order_electrode, f_interpolate, sig):
    #calculates order of atlas and takes care of duplicates/noises
    #currently, for real time, there aren't always 16 clear sources, after a certain threshold they are assigned '-1' and might not show

    # the potential electrode assignemnts
    unique = np.unique(order_electrode)
    potential_electrodes = np.setdiff1d(np.arange(16), unique)

    # the flattened coordinates array
    flat_coordinates = np.ravel_multi_index((y_coor, x_coor), f_interpolate[0].shape)

    # go over duplicates of electrodes assignments and decide which one has noise and which doesn't by the amount of red
    # start with threshold of 0.75 and go down if nececary

    # priority for the signifcant electrodes
    unique, counts = np.unique(order_electrode[sig], return_counts=True)
    duplicates = unique[counts > 1]

    potential_electrodes, order_electrode = atlas_duplicates_sig(duplicates, potential_electrodes, order_electrode,
                                                                 f_interpolate, flat_coordinates, sig)

    # next do the same procedure for all other electrodes
    unique, counts = np.unique(order_electrode, return_counts=True)
    duplicates = unique[counts > 1]

    potential_electrodes, order_electrode = atlas_duplicates_not_sig(duplicates, potential_electrodes,
                                                                     order_electrode, f_interpolate, flat_coordinates,
                                                                     sig)

    # insert the potential electrodes in the order_electrodes array just to order it correctly
    order_electrode[np.where(order_electrode == -1)[0]] = potential_electrodes


    array_indices = np.argsort(order_electrode)[wanted_order]

    return array_indices, order_electrode, potential_electrodes

# Function to handle checkbox selection
def update_visibility(axs, index):
    visibility = axs[index].get_visible()  # Get the current visibility state of the first subplot
    axs[index].set_visible(not visibility)  # Toggle the visibility of the first subplot (the signal)
    axs[index+1].set_visible(not visibility)  # Toggle the visibility of the subplot next to it (the heatmap)

    #axs[index].set_visible(not axs[index].get_visible())
    plt.draw()

def plot_sources(potential_electrodes, f_interpolate, array_indices, start_time, end_time, Y, sig_convert):
    # heatmaps with sources
    row_num, col_num = 4, 12
    ratio = round(height / width, 1)

    fig, axes = plt.subplots()
    spec = fig.add_gridspec(row_num, col_num)
    #fig.set_size_inches(1 * col_num, 1 * ratio * row_num)
    # Autoscale the figure to fit the screen
    fig.set_dpi(300)

    axs = []
    for i in range(row_num):
        for j in range(col_num):
            if (j == 2 or j == 5 or j == 8 or j == 11):
                ax = fig.add_subplot(spec[i, j])
                axs.append(ax)
            elif (j == 0 or j == 3 or j == 6 or j == 9):
                ax = fig.add_subplot(spec[i, j:j + 2])
                axs.append(ax)
    axes.axis('off')

    source = 0
    for j in range(len(axs)):
        source = int(j / 2)
        if (j % 2 != 0):
            axs[j].imshow(image)
            axs[j].pcolormesh(f_interpolate[array_indices[source]], cmap='jet', alpha=0.5)
            axs[j].set_aspect('auto')
            axs[j].axis('off')
        else:
            duration= end_time - start_time
            axs[j].plot(np.arange(start_time, end_time), Y[array_indices[source], -duration:], color=colormap.colors[0], lw=0.5)
            axs[j].margins(0)
            axs[j].set_ylim([-5, 5])
            axs[j].set_title('source %d' % (wanted_order[source] + 1), fontsize=7)

        # only show the y-axis for the left-most column
        if (j % 8 == 0):
            axs[j].tick_params(axis='y', labelsize=5, direction='in', length=4, width=1)
        else:
            axs[j].tick_params(axis='y', left=False, labelleft=False)

        # only show the x-axis for the bottom-most column
        if (j >= 24):
            axs[j].tick_params(axis='x', labelsize=5, direction='in', length=4, width=1)
        else:
            axs[j].tick_params(axis='x', bottom=False, labelbottom=False)




        time_func = (lambda a: divmod(int(a / sampling_rate), 60))
        ticks = ticker.FuncFormatter(
            lambda x, pos: (str(time_func(x)[0]).zfill(2) + ':' + str(time_func(x)[1]).zfill(2)))
        axs[j].xaxis.set_major_formatter(ticks)

    global checkboxes, checkbox_vars





    # Create a list to store the IntVar variables for checkbox states
    checkbox_vars = []
    checkboxes=[]

    # Create checkboxes for each subplot
    for i in range(0,len(axs), 2):
        source = int(i/ 2)

        var = tk.IntVar()

        if(wanted_order[source] in sig_convert):  # sig subplots are selected by default
            var.set(1)
            axs[i].set_visible(True)
            axs[i+1].set_visible(True)
        else:
            var.set(0)
            axs[i].set_visible(False)
            axs[i+1].set_visible(False)

        checkbox = tk.Checkbutton(window, text='Source %d'%(wanted_order[source] + 1), variable=var,
                                  command=lambda i=i: update_visibility(axs, i))
        #checkbox.select()  # Initially select all checkboxes

        # Customize the appearance of the checkbox text
        if(wanted_order[source] in sig_convert):
            checkbox.config(
                bg='yellow',  # Set yellow background
                fg='black',  # Set font color
                font=('Arial', 12, 'bold')  # Set font style (Arial, size 12, bold)
            )

        if(wanted_order[source]  in potential_electrodes):
            checkbox.config(
                fg='grey',  # Set green font color
                font=('Arial', 5)  # Set font style
            )

        checkbox.grid(row=int(source/4), column=source%4)

        checkbox_vars.append(var)
        checkboxes.append(checkbox)

    #fig.subplots_adjust(hspace=0.8)
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.95, hspace=0.5)

    plt.show()


def get_time():
    time_value = int(entry_time.get())


def get_input():

    if(entry_time.get()):
        ica_time=int(entry_time.get())*sampling_rate
    else:
        ica_time=None


    global button_pressed
    if button_pressed:
        # Clear previous checkboxes
        for checkbox in checkboxes:
            checkbox.destroy()
    else:
        #print("Button was pressed for the first time.")
        button_pressed = True

    # Retrieve the value entered by the user
    input_value = int(entry.get())
    # Perform any desired actions with the input value
    print("Input value:", input_value)
    Y, K, W, sort_intensity, relative_intensity, start_time, end_time = obtain_sources_for_segement(input_value, ica_time)
    f_interpolate, order_electrode, sig = calculate_heatmaps(relative_intensity, sort_intensity, W, K)
    array_indices, order_electrode, potential_electrodes=calculate_atlas(order_electrode, f_interpolate, sig)
    sig_convert=wanted_order[np.where(np.isin(array_indices, sig))[0]]
    plot_sources(potential_electrodes, f_interpolate, array_indices, start_time, end_time, Y, sig_convert)

    print('success')



# Create the main window
window = tk.Tk()

button_pressed = False

# Create an entry field
entry = tk.Entry(window)
entry.grid(row=5, column=0)

# Create a button to get the input value
button = tk.Button(window, text="Submit", command=get_input)
button.grid(row=6, column=0)



def button_click(text):
    # Handle the button click event here

    #will be self in real time
    global offset_before
    global offset_after

    print(text)
    #current_time = datetime.now().time()
    times[text]=datetime.now()
    if(text=="Start Expression"):
        offset_before=(times["Start Expression"] - times["Start Relax"]).total_seconds()
        print(offset_before)
    elif(text=="End Relax"):
        offset_after = (times["End Relax"]-times["End Expression - Relax Again"]).total_seconds()
        print(offset_after)


    # Perform any desired actions with the input value

button_texts = ["Start Relax", "Start Expression", "End Expression - Relax Again", "End Relax"]

for index, text in enumerate(button_texts):
    button = tk.Button(window, text=text, command=lambda t=text: button_click(t))
    button.grid(row=7+index, column=0)


# Create an second entry field - for accumulation time
entry_time = tk.Entry(window)
entry_time.grid(row=11, column=0)

# Create a button to get the input value
button = tk.Button(window, text="Submit", command=get_time)
button.grid(row=12, column=0)

# Start the Tkinter event loop
window.mainloop()