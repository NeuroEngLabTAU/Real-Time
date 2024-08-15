import cv2
import datetime
import threading
import os

# function for video recording
def record_video(webcam, fourcc, is_recording_video, saving_path):

    # Get the current time including milliseconds
    recording_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")[:-3]
    output_filename = os.path.join(saving_path,f"video_{recording_start_time}.avi")
    print(f"Video recording started at {recording_start_time} ...")

    # Define the video writer object
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

    # Start recording until user interrupts or is_recording is False
    while is_recording_video:
        ret, frame = webcam.read()  # Read frame from webcam

        if ret:
            # Write the frame to the output video file
            out.write(frame)
        else:
            print("Failed to capture frame")
            break

    print("Video recording stopped at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    # Release the video writer object
    out.release()


# functions to control video recording
def start_video(cap, is_recording_video, fourcc, saving_path):
    # global is_recording_video
    fourcc = fourcc
    if not is_recording_video:
        is_recording_video = True
        record_thread = threading.Thread(target=record_video, args=(cap, fourcc, is_recording_video, saving_path))
        record_thread.start()
        # data.add_annotation('started video recording')
        print(is_recording_video)
    return is_recording_video


def stop_video(is_recording_video):
    print("entered stop_video")
    # global is_recording_video
    if is_recording_video:
        print("entered if")
        is_recording_video = False
        print(is_recording_video)
        # data.add_annotation('stopped video recording')

