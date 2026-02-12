# Real-Time Eye Tracking Project
February 2026

## Usage and requirements
### Setting up environment
- Make a virtual environment and run `pip install -r requirements.txt`
- Download the pre-trained Google MediaPipe face landmarker [model file](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) and place it in `resources/`
- Run `main.py`

### Calibration
If everything is working correctly, the first thing you will see is the calibration window with a 9x9 grid. You can adjust the size of this window in the configuration at the top of `main.py`. Alternatively, you may want to open the window in fullscreen mode. Your webcam will also be activated. Make sure you have picked the correct source, which you can also adjust in the configuration.

A red dot will be displayed to indicate which of the 9 dots to look at. Make sure to keep your head as still as possible; only move your eyes. The calibration sample will only be taken when you press the space bar on each highlighted dot. When you press the space bar the final time, the window should close and be replaced by a fullscreen window of the main eye tracking system.

### Usage instructions
As before, keep your head as still as possible and aim to look within the calibration area. You should see a blurry cursor move around the window to match your eye movement that changes in size and colour to reflect the speed of your movement.


## Problem statement
The aim of this project is to create a proof-of-concept for a real-time, calibration-based eye tracking system.

## System overview

## Results and limitations
From start to finish, this project took about a week to complete.

## Future work

### References
- [MediaPipe example Jupyter Notebook](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=a49D7h4TVmru)
