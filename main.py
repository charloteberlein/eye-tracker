# created 04.02.2026
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as v
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
import threading


## GLOBALS
VIDEO_FEED = 1 # input camera
W, H = 3024, 1964 # screen dimensions
# default landmark indices within mesh
LEFT_EYE_LANDMARKS = (463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,
                      373, 374, 380, 381, 382, 362)
RIGHT_EYE_LANDMARKS = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154,
                       153, 145, 144, 163, 7)
LEFT_IRIS_LANDMARKS = (473, 474, 475, 477, 476)
RIGHT_IRIS_LANDMARKS = (468, 469, 470, 471, 472)
# specific landmarks to use
EYE_CORNERS_L, EYE_CORNERS_R = (463, 263), (133, 33) # (inner, outer)
EYE_APEX_L, EYE_APEX_R, PUPILS = (386, 145), (159, 374), (473, 468)
LANDMARKS = EYE_CORNERS_L + EYE_CORNERS_R + EYE_APEX_L + EYE_APEX_R + PUPILS

R = None # current face result
def callback_func(out, im, time) -> None:
    global R
    R = out
pupil_bounds = None # modify after calibration
processing = False

OPTIONS = v.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(
            model_asset_path="resources/face_landmarker.task"
        ),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=callback_func
    )


def get_norm_pupil_position(F) -> tuple[tuple[float, float], tuple[float, float]]:
    # get all coords for both eyes
    lx = np.array([F[i].x for i in LEFT_EYE_LANDMARKS])
    ly = np.array([F[i].y for i in LEFT_EYE_LANDMARKS])
    rx = np.array([F[i].x for i in RIGHT_EYE_LANDMARKS])
    ry = np.array([F[i].y for i in RIGHT_EYE_LANDMARKS])

    # normalise pupil coords within bounding box
    lx_norm = np.clip((F[PUPILS[0]].x-lx.min()) / (lx.max()-lx.min()), 0.0, 1.0)
    ly_norm = np.clip((F[PUPILS[0]].y-ly.min()) / (ly.max()-ly.min()), 0.0, 1.0)
    rx_norm = np.clip((F[PUPILS[1]].x-rx.min()) / (rx.max()-rx.min()), 0.0, 1.0)
    ry_norm = np.clip((F[PUPILS[1]].y-ry.min()) / (ry.max()-ry.min()), 0.0, 1.0)
    return ((lx_norm, ly_norm), (rx_norm, ry_norm))


def view_eye_position() -> NDArray | None:
    global OPTIONS, pupil_bounds, processing
    processing = True

    # load pre-trained face landmarker model from mediapipe
    with v.FaceLandmarker.create_from_options(OPTIONS) as landmarker:
        # get pictures from webcam
        video = cv2.VideoCapture(VIDEO_FEED)
        start_time = datetime.now()

        total = np.array([[0.0,0.0], [0.0,0.0]])

        while True:
            ret, frame = video.read()
            if not ret:
                raise Exception("Video Source Error")

            # calculate time in ms
            timedelta = datetime.now()-start_time
            timestamp_ms = int(timedelta.total_seconds()*1000)
            if timestamp_ms >= 200:
                break

            # detect face landmarks
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(im, timestamp_ms)

            if R and R.face_landmarks:
                for face_landmarks in R.face_landmarks:
                    r = get_norm_pupil_position(face_landmarks)
                    if r is None:
                        raise ValueError
                    total += r # implicit element-wise addition
        
        video.release()
        pupil_bounds = np.clip(total/len(total), 0.0, 1.0)
        processing = False


def calibrate(dots) -> None:
    global processing, pupil_bounds
    curr_dot = -1
    mean_pupil_positions = []

    # plot calibration rig
    while True:
        # make fullscreen grey background
        canvas = np.full((H,W,3), (128,128,128), np.uint8)

        # draw on dots!
        if curr_dot > -1:
            for dot in dots:
                cv2.circle(canvas, dot, 12, (50, 50, 50), -1)
            cv2.circle(canvas, dots[curr_dot], 16, (0,0,255), -1)

        cv2.imshow("Calibration", canvas)

        # quit or progress
        k = cv2.waitKey(1)
        if k & 0xFF in (ord('q'), 27): # escape or q to quit prematurely 
            break
        elif k == 32 and not processing: # space bar to progress calibration pt
            if curr_dot < 8:
                try:
                    threading.Thread(target=view_eye_position).start()
                except Exception as e:
                    print(e)
                curr_dot += 1
            else:
                mean_pupil_positions.append(pupil_bounds)
                print(mean_pupil_positions)
        elif processing:
            if pupil_bounds is not None:
                mean_pupil_positions.append(pupil_bounds)
                pupil_bounds = None

    cv2.destroyAllWindows()


def start():
    # compute dot positions based on screen dimensions
    w, h = 0.95*W, 0.95*H
    dot_positions = []
    for i in range(3):
        for j in range(3):
            x_pos, y_pos = int(j*w/2 + w*0.025), int(i*h/2 + h*0.025)
            dot_positions.append((x_pos, y_pos))
    calibrate(dot_positions)


def main():
    start()
    # add main program code here


if __name__ == "__main__":
    main()


"""
[array([[1.        , 0.69877229],
       [0.8240079 , 0.69121015]]), array([[1.        , 0.9288538 ],
       [0.93730834, 0.91836238]]), array([[1.        , 1.        ],
       [0.76083955, 0.99219775]]), array([[1.        , 0.9780449 ],
       [1.        , 0.95482915]]), array([[1.        , 0.97544361],
       [0.93642384, 0.93766639]]), array([[1.        , 0.99435151],
       [0.73315262, 0.97166966]]), array([[1.        , 0.93394394],
       [0.97971994, 0.91240782]]), array([[1.        , 0.91029546],
       [0.90037219, 0.89745984]])]
"""