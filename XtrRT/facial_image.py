import os
import cv2
import numpy as np
from matplotlib import image as mpimg
from matplotlib.path import Path

# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
# Construct the image path relative to the script directory
IMAGE_PATH = os.path.join(script_directory, "face-muscles-anatomy.jpg")
X_COOR = [557, 398, 336, 444, 466, 342, 389, 490, 601, 450, 335, 328, 422, 545, 551, 689]
Y_COOR = [836, 786, 690, 721, 657, 634, 586, 599, 567, 541, 537, 477, 387, 381, 290, 289]


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


#######################################
###### select electrode location ######
#######################################

def click_event(event, x, y, flags, params):
    global X_COOR
    global Y_COOR
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        X_COOR.append(x)
        Y_COOR.append(y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '(' + str(x) + ',' +
                    str(y) + ')', (x, y), font,
                    0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

    return X_COOR, Y_COOR


def get_location():
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


if len(X_COOR) == 0:  # if manually inserted coordinates, there is not need to get the electrode locations (otherwise select electrodes from image)
    get_location()


# create a dummy heatmap
def fill_polygon(vertices, num_rows, num_cols):
    # Create a grid of points
    x = np.linspace(0, num_cols - 1, num_cols)
    y = np.linspace(0, num_rows - 1, num_rows)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.flatten(), yy.flatten())).T

    # Create a path from the polygon vertices
    path = Path(vertices)

    # Determine whether each point is inside the polygon
    mask = path.contains_points(points)
    mask = mask.reshape(num_rows, num_cols)

    # Fill the grid with random values between 0 and 1, and NaN values outside the polygon
    result = np.empty((num_rows, num_cols))
    result[~mask] = np.nan
    result[mask] = np.random.rand(np.sum(mask))

    return result


def get_dummy_heatmap(height,width):
    # create dummy heatmap for funcAnimator setup
    vertices = list(zip(X_COOR, Y_COOR))
    num_rows, num_cols = height, width
    d_interpolate = fill_polygon(vertices, num_rows, num_cols)
    return d_interpolate


