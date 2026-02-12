# created 04.02.2026
# vision modules
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as v
# computation modules
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# plotting modules
import matplotlib.pyplot as plt


# global constants
VIDEO_FEED = 1 # input camera
W, H = 3024, 1964 # screen dimensions
CENTRE = np.array([W//2, H//2]) # screeen centre coordinates
SMOOTHING = (0.6,0.8) # smoothing strength coefficient
# default landmark indices within mesh
LEFT_EYE_LANDMARKS = (463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,
                      373, 374, 380, 381, 382, 362)
RIGHT_EYE_LANDMARKS = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154,
                       153, 145, 144, 163, 7)
LEFT_IRIS_LANDMARKS = (473, 474, 475, 477, 476)
RIGHT_IRIS_LANDMARKS = (468, 469, 470, 471, 472)
# specific landmarks to use in this project
EYE_CORNERS_L, EYE_CORNERS_R = (463, 263), (133, 33) # (inner, outer)
EYE_APEX_L, EYE_APEX_R, PUPILS = (386, 145), (159, 374), (473, 468)
LANDMARKS = EYE_CORNERS_L + EYE_CORNERS_R + EYE_APEX_L + EYE_APEX_R + PUPILS

# global variables
R = None # current face result
processing = False

# helper and callback functions
def callback_func(out, im, time) -> None: # required for mediapipe landmarker
    global R
    R = out
def pt(p): # coordinate arrays -> tuples
    return int(p[0]), int(p[1])

# mediapipe facial landmarker options & filepath
OPTIONS = v.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(
            model_asset_path="resources/face_landmarker.task"
        ),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=callback_func
    )


def get_pupil_features(F) -> NDArray[np.float64]:
    """
    Returns positions of the eye pupil centres relative to the centre of each
    eye.

    args:
        F:          face landmarks array

    returns:
        lpx, lpy:   left pupil position relative to left eye centre
        rpx, rpy:   right pupil position relative to right eye centre
    """
    # left
    li, lo = EYE_CORNERS_L # inner & outer
    lx0, ly0 = F[li].x, F[li].y
    lx1, ly1 = F[lo].x, F[lo].y
    lp = F[PUPILS[0]]
    l_width = np.hypot(lx1-lx0, ly1-ly0) + 1e-6

    lcx, lcy = (lx0+lx1)/2, (ly0+ly1)/2 # find left eye centre
    lpx = (lp.x-lcx) / l_width
    lpy = (lp.y-lcy) / l_width

    # right
    ri, ro = EYE_CORNERS_R
    rx0, ry0 = F[ri].x, F[ri].y
    rx1, ry1 = F[ro].x, F[ro].y
    rp = F[PUPILS[1]]

    r_width = np.hypot(rx1-rx0, ry1-ry0) + 1e-6
    rcx, rcy = (rx0+rx1)/2, (ry0+ry1)/2 # find right eye centre
    rpx = (rp.x-rcx) / r_width
    rpy = (rp.y-rcy) / r_width

    return np.array([lpx,lpy,rpx,rpy], dtype=np.float64)


def view_eye_position() -> NDArray | None:
    """
    Detects eye features using MediaPipe face landmarks.

    returns:
        median of features array
    """
    global OPTIONS, processing
    processing = True

    features = []

    # load pre-trained face landmarker model from mediapipe
    with v.FaceLandmarker.create_from_options(OPTIONS) as landmarker:
        # get pictures from webcam
        video = cv2.VideoCapture(VIDEO_FEED)
        start_time = datetime.now()

        while True:
            ret, frame = video.read()
            if not ret:
                raise Exception("Video Source Error")

            # calculate time in ms
            timedelta = datetime.now()-start_time
            timestamp_ms = int(timedelta.total_seconds()*1000)
            if timestamp_ms >= 300:
                break

            # detect face landmarks
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(im, timestamp_ms)

            if R and R.face_landmarks:
                for face_landmarks in R.face_landmarks:
                    features.append(get_pupil_features(face_landmarks))
        
        video.release()

    processing = False
    if features:
        return np.mean(features, axis=0)
    return None


def calibrate(dots) -> tuple[NDArray, NDArray]:
    """
    Manages calibration procedure.
    
    args:
        dots (array):   list of calibration point coordinates
    
    returns:
        X:              features
        Y:              corresponding point coordinates
    """
    curr_dot = 0
    X, Y = [], []

    # plot calibration rig
    while curr_dot < len(dots):
        # make fullscreen grey background
        canvas = np.full((H,W,3), (128,128,128), np.uint8)

        # draw on dots!
        for dot in dots:
            cv2.circle(canvas, dot, 12, (50, 50, 50), -1)
        cv2.circle(canvas, dots[curr_dot], 16, (0,0,255), -1)

        cv2.imshow("Calibration", canvas)

        # quit or progress
        k = cv2.waitKey(1)
        if k & 0xFF in (ord('q'), 27): # escape or q to quit prematurely 
            break
        elif k == 32 and not processing: # space bar to progress calibration pt
            feat = view_eye_position()
            if feat is not None:
                X.append(feat)
                Y.append(dots[curr_dot])
                curr_dot += 1

    cv2.destroyAllWindows()
    return np.array(X), np.array(Y)


def train_models(X, Y) -> tuple:
    """
    Fits a bilinear regression model to calibration data, using feature scaling
    to improve fit.

    args:
        X:      features
        Y:      calibration point screen coordinates
    
    returns:
        models: linear regression models for x and y
    """
    model_x = make_pipeline(
        StandardScaler(),
        Ridge()
    )
    model_y = make_pipeline(
        StandardScaler(),
        Ridge()
    )

    model_x.fit(X, Y[:,0])
    model_y.fit(X, Y[:,1])

    return model_x, model_y


def predict_gaze(models, feats, prev) -> tuple[int, int]:
    """
    Predicts a screen coordinate relative to the pupil position.

    args:
        models: bilinear regression model?
        feats:  pupil features
        prev:   previous gaze coordinates
    
    returns:
        x, y:   screen coordinates
    """
    model_x, model_y = models

    x = model_x.predict([feats])[0]
    y = model_y.predict([feats])[0]

    # smoothing
    x = SMOOTHING[0]*x + (1-SMOOTHING[0])*prev[0]
    y = SMOOTHING[1]*y + (1-SMOOTHING[1])*prev[1]

    return int(np.clip(x, 0.05*W, 0.95*W)), int(np.clip(y, 0.05*H, 0.95*H))


def start() -> NDArray:
    """
    Initialises calibration procedure.

    returns:
        dot_positions:  calibration point screen coordinates
    """
    # compute dot positions based on screen dimensions
    w, h = 0.95*W, 0.95*H
    dot_positions = []
    for i in range(3):
        for j in range(3):
            x_pos, y_pos = int(j*w/2 + w*0.025), int(i*h/2 + h*0.025)
            dot_positions.append((x_pos, y_pos))
    return np.array(dot_positions)


def main() -> None:
    dot_pos = start()
    X, Y = calibrate(dot_pos)
    models = train_models(X,Y)
    prev_coords = CENTRE

    # load pre-trained face landmarker model from mediapipe
    with v.FaceLandmarker.create_from_options(OPTIONS) as landmarker:
        # get pictures from webcam
        video = cv2.VideoCapture(VIDEO_FEED)
        start_time = datetime.now()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # calculate time in ms
            timedelta = datetime.now()-start_time
            timestamp_ms = int(timedelta.total_seconds()*1000)

            # detect face landmarks
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(im, timestamp_ms)

            # draw calculated eye position on screen
            canvas = np.full((H,W,3), (128,128,128), np.uint8)
 
            if R and R.face_landmarks:
                for face_landmarks in R.face_landmarks:
                    feats = get_pupil_features(face_landmarks)
                    gaze = predict_gaze(models, feats, prev_coords)
                    # plot velocity (to quantify confidence for later)
                    velocity = np.gradient(np.column_stack((gaze, prev_coords)))
                    cv2.putText(canvas, str(abs(np.hypot(velocity[1][0], velocity[1][1]))), (100,100), 0, 1, 0)
                    cv2.line(canvas, gaze, tuple(prev_coords), (255,0,0), 12)

                    prev_coords = gaze
            
            cv2.imshow("Real-Time Gaze", canvas)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): # esc or q to quit
                break
        
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()