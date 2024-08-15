import threading
import cv2
import datetime
import os

# Function for video recording
def record_video(webcam, fourcc, stop_event, saving_path):
    # Get the current time including milliseconds
    recording_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")[:-3]
    output_filename = os.path.join(saving_path, f"video_{recording_start_time}.avi")
    print(f"Video recording started at {recording_start_time} ...")

    # Define the video writer object
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

    # Start recording until the stop event is set
    while not stop_event.is_set():
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


# Functions to control video recording
def start_video(cap, stop_event, fourcc, saving_path):
    if not stop_event.is_set():
        stop_event.clear()  # Ensure the stop event is clear
        record_thread = threading.Thread(target=record_video, args=(cap, fourcc, stop_event, saving_path))
        record_thread.start()
        print("Recording started")


def stop_video(stop_event):
    print("entered stop_video")
    if not stop_event.is_set():
        stop_event.set()  # Signal the recording thread to stop
        print("Recording stopped")

