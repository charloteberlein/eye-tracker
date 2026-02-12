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
processing = False # flag for calibration async
video = None # prevent cv2 memory leak when closing mpl window

# helper and callback functions
def callback_func(out, im, time) -> None: # required for mediapipe landmarker
    global R
    R = out

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
    lp = F[PUPILS[0]]

    li, lo = EYE_CORNERS_L # inner & outer
    lx0, ly0 = F[li].x, F[li].y
    lx1, ly1 = F[lo].x, F[lo].y
    l_width = np.hypot(lx1-lx0, ly1-ly0) + 1e-6

    lu, ld = EYE_APEX_L
    lx2, ly2 = F[lu].x, F[lu].y
    lx3, ly3 = F[ld].x, F[ld].y
    l_height = np.hypot(lx3-lx2, ly3-ly2) + 1e-6

    lcx, lcy = (lx0+lx1)/2, (ly0+ly1)/2 # find left eye centre
    lpx = (lp.x-lcx) / l_width
    lpy = (lp.y-lcy) / l_height

    # right
    rp = F[PUPILS[1]]

    ri, ro = EYE_CORNERS_R
    rx0, ry0 = F[ri].x, F[ri].y
    rx1, ry1 = F[ro].x, F[ro].y
    r_width = np.hypot(rx1-rx0, ry1-ry0) + 1e-6

    ru, rd = EYE_APEX_R
    rx2, ry2 = F[ru].x, F[ru].y
    rx3, ry3 = F[rd].x, F[rd].y
    r_height = np.hypot(rx3-rx2, ry3-ry2) + 1e-6

    rcx, rcy = (rx0+rx1)/2, (ry0+ry1)/2 # find right eye centre
    rpx = (rp.x-rcx) / r_width
    rpy = (rp.y-rcy) / r_height

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


def make_confidence_gradient(r=100, c=(0,0,0), alpha=2.0):
    # set up coordinate grid, mask, and gradient for cursor blob
    y, x = np.ogrid[-r:r,-r:r]
    max_radius = np.clip(np.hypot(x,y)/r, 0, 1)
    alpha = (1-max_radius)**alpha

    # make img
    img = np.zeros((r*2, r*2, 4))
    img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3] = c[0], c[1], c[2], alpha

    return img


def on_fig_exit(event):
    global video # ensure proper memory handling
    if (event.name == "close_event" or (event.name == "key_press_event" and
                                        event.key == "escape")):
        if (event.name == "key_press_event" and event.key == "escape"):
            plt.close(event.canvas.figure)
        if video is not None: video.release()
        cv2.destroyAllWindows()
        quit()


def main() -> None:
    global video
    dot_pos = start()
    X, Y = calibrate(dot_pos)
    models = train_models(X,Y)
    prev_coords = CENTRE

    plt.ion() # turn on interactive mode for Esc event connection

    # set up plot structure
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Eye Tracking", fontweight="bold")
    plt.text(0.99, 0.01,"© Charlot Eberlein 2026",c=(0.5,0.5,0.5),fontsize=10,
         ha='right',va='bottom',transform=plt.gca().transAxes)
    plt.gca().set_facecolor('#f0f0f0')
    ax.set_xlim(0,W)
    ax.set_ylim(0,H)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # get colourmap
    cmap = plt.get_cmap("Blues_r")

    # init gradient image
    gradient_im = make_confidence_gradient(c=cmap(0.0))
    ax_im = ax.imshow(gradient_im, extent=(0,0,0,0))
    fig.tight_layout()

    manager = plt.get_current_fig_manager()
    if manager is not None: manager.full_screen_toggle()

    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.canvas.mpl_connect("close_event", on_fig_exit)
    fig.canvas.mpl_connect("key_press_event", on_fig_exit)

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
            if R and R.face_landmarks:
                for face_landmarks in R.face_landmarks:
                    feats = get_pupil_features(face_landmarks)
                    gaze = predict_gaze(models, feats, prev_coords)
                    # plot velocity (clipped to [200, 1200])
                    V = 2*np.mean(np.clip(np.gradient(np.column_stack(
                        (gaze, prev_coords))), 100, 600))
                    c0, c1, c2, _ = cmap(np.clip((V-200)/500, 0, 0.7))

                    ax_im.set_extent((gaze[0]-V, gaze[0]+V,
                                      H-gaze[1]+V, H-gaze[1]-V)) # type: ignore
                    new_im = gradient_im.copy()
                    new_im[:,:,0], new_im[:,:,1], new_im[:,:,2] = c0, c1, c2
                    ax_im.set_data(new_im)

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                    prev_coords = gaze
        
        # this is bypassed if mpl figure (window) closed
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()