import datetime
import cv2
import subprocess

def show_image_opencv(frame, windowns_name):
    cv2.imshow(windowns_name, frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        return -1
    return None


def video_frame_reader(url, queue, block=False):
    # initialize the video stream, pointer to output video file, and frame dimensions
    video_capture = cv2.VideoCapture(url)

    # keep looping infinitely until the thread is stopped
    _start_time = datetime.datetime.now()
    global_frame_counter = 0
    frame_counter = 0
    n_missed_frames = 0
    while True:
        # otherwise, read the next frame from the stream
        (_, frame) = video_capture.read()
        global_frame_counter += 1
        frame_counter += 1
        try:
            queue.put(frame, block=block)
        except:
            n_missed_frames += 1

        if frame_counter % 100 == 0:
            _elapsed = (datetime.datetime.now() - _start_time).total_seconds()
            estimated_fps = frame_counter / _elapsed
            frame_counter = 0
            _start_time = datetime.datetime.now()
            print("Webcam estimated fps: %s - Missed frames=%s" % (estimated_fps, n_missed_frames))
