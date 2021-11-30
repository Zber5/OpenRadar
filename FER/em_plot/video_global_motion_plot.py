import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import skvideo.datasets

import skvideo

skvideo.setFFmpegPath('C:/Users/Zber/Documents/ffmpeg/bin')

import skvideo.io
import skvideo.motion
import cv2

# it is not working now

path1 = "C:/Users/Zber/Desktop/Emotion_face/Anger_0/Anger_0_00019.jpg"
path2 = "C:/Users/Zber/Desktop/Emotion_face/Anger_0/Anger_0_00033.jpg"

frame1 = cv2.imread(path1)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame2 = cv2.imread(path2)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


motion = skvideo.motion.globalEdgeMotion(frame1, frame2)

print("")