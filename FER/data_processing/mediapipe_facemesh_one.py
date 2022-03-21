import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import time
from pathlib import Path
import numpy as np
import math
from typing import List, Mapping, Optional, Tuple, Union
from imutils.video import count_frames
import os
import enum

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

frame_id = 0
camera_fps = 30
width = 1280
height = 720
dims = 2

from mediapipe.python.solutions.drawing_utils import DrawingSpec
import enum
from copy import deepcopy

_THICKNESS_DOT = 1
_RADIUS = 2

BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
GRAY_COLOR = (128, 128, 128)
CYAN_COLOR = (255, 255, 0)
CORAL_COLOR = (80, 127, 255)
BROWN_COLOR = (96, 164, 244)

FACE_LANDMARK = dict(
    lips=(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178,
          87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415),
    left_eye=(263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398),
    left_eyebrow=(276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    right_eye=(33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173),
    right_eyebrow=(46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    face_oval=(10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
               149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109),
    nose=(1, 2, 98, 327),
    right_cheek=(205,),
    left_check=(425,),
    midway=(168,),
    # jaw=(32, 146, 176, 208, 171, 148, 199, 175, 152, 428, 396, 377, 282, 369, 400),
    jaw=(200, 199, 175),
)


def _others_generator():
    others = []
    ex = []
    for i in FACE_LANDMARK:
        ex += FACE_LANDMARK[i]

    for index in range(468):
        if index not in ex:
            others.append(index)
    return tuple(others)


def get_key_flm(is_all=False):
    key_flm = deepcopy(FACE_LANDMARK)
    if is_all:
        others = _others_generator()
        key_flm['others'] = others
    return key_flm


def get_face_landmark_style():
    """Returns the default hand landmark drawing style.

    Returns:
        A mapping from each hand landmark to the default drawing spec.
    """
    key_flm_style = get_key_flm(is_all=True)
    style = {
        key_flm_style['lips']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['left_eye']:
            DrawingSpec(
                color=BLACK_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 1),
        key_flm_style['left_eyebrow']:
            DrawingSpec(
                color=BROWN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['right_eye']:
            DrawingSpec(
                color=BLACK_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 1),
        key_flm_style['right_eyebrow']:
            DrawingSpec(
                color=BROWN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['face_oval']:
            DrawingSpec(
                color=BLUE_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        key_flm_style['jaw']:
            DrawingSpec(
                color=CYAN_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['nose']:
            DrawingSpec(
                color=CORAL_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['right_cheek']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 3),
        key_flm_style['left_check']:
            DrawingSpec(
                color=RED_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS + 3),
        key_flm_style['midway']:
            DrawingSpec(
                color=GRAY_COLOR, thickness=_THICKNESS_DOT + 1, circle_radius=_RADIUS + 1),
        key_flm_style['others']:
            DrawingSpec(
                color=GREEN_COLOR, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    }

    face_landmark_style = {}
    for k, v in style.items():
        for landmark in k:
            face_landmark_style[landmark] = v
    return face_landmark_style


# def render_color
def get_num_flm():
    pos = []
    start = 0
    key_flm = get_key_flm()
    for key in key_flm:
        length = len(key_flm[key])
        end = start + length
        pos.append((start, end))
        start = end
    return start, pos


def flm_score(all_flm):
    num_flm, flm_seg = get_num_flm()
    pre = None
    key_flm = get_key_flm()

    all_dis = np.zeros((all_flm.shape[0] - 1, num_flm))

    for fid, flm in enumerate(all_flm):
        cur = np.zeros((num_flm, dims))
        for key, s_e in zip(key_flm, flm_seg):
            indices = np.array(key_flm[key])
            start, end = s_e
            pos = flm[indices]
            cur[start:end] = pos
        if pre is not None:
            # calculate the score
            dist = np.linalg.norm(cur - pre, axis=1)
            all_dis[fid - 1] = dist
        pre = cur

    return all_dis


def distance(all_flm, normalise=True):

    # landmark normalization
    nose_index=33
    if normalise:
        for frame_index in range(all_flm.shape[0]):
            nose_lm = all_flm[frame_index, nose_index]
            all_flm[frame_index] = all_flm[frame_index] - nose_lm

    all_dis = np.zeros((all_flm.shape[0] - 1, all_flm.shape[1]))

    pre = None
    for fid, cur in enumerate(all_flm):
        if pre is not None:
            # calculate the score
            dist = np.linalg.norm(cur - pre, axis=1)
            all_dis[fid - 1] = dist
        pre = cur

    return all_dis


def key_average_score(scores, num_frame=89):
    num_flm, flm_seg = get_num_flm()
    num_keys = len(flm_seg)
    data = np.zeros((num_frame, num_keys))
    for i in range(num_keys):
        s_ind, e_ind = flm_seg[i]
        flmscore = scores[:, s_ind:e_ind]
        mean = np.mean(flmscore, axis=1)
        length = np.shape(flmscore)[0]
        if length <= num_frame:
            data[:length, i] = mean
        else:
            data[:num_frame, i] = mean
    return data


def flm_detector(video_path, output_path, output_as_video=False, output_flm_video=False, output_flm_npy=False, dim=2):
    cv_plot = False
    total_FLms = 468

    # face mesh settings
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    # drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    drawSpec = get_face_landmark_style()
    black_drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    connection_drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    frame_id = 0
    pTime = 0

    cap = cv2.VideoCapture(video_path)
    total_frame = count_frames(video_path)
    all_FLms = np.zeros((total_frame, total_FLms, dim))
    if output_as_video or output_flm_video:
        # video_output = os.path.join(output_path,
        #                             "{}_landmark_flm.avi".format(os.path.basename(os.path.normpath(output_path))))
        video_output = os.path.join(output_path,
                                    "{}_landmark.avi".format(os.path.basename(os.path.normpath(output_path))))
        # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       camera_fps,
                                       (width, height))

    while True:
        success, img = cap.read()
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not success:
            # print("Finish")
            break
        imgRGB = img
        results = faceMesh.process(imgRGB)
        black_img = np.zeros(img.shape)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, connection_drawSpec)
                mpDraw.draw_landmarks(img, faceLms, None, drawSpec, connection_drawSpec)
                mpDraw.draw_landmarks(black_img, faceLms, None, black_drawSpec, None)

            np_faceLms = np.zeros((total_FLms, 2))
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = _normalized_to_pixel_coordinates(lm.x, lm.y, iw, ih)
                # x,y = int(lm.x*iw), int(lm.y*ih)
                # print(id,x,y)
                np_faceLms[id] = [x, y]

            all_FLms[frame_id] = np_faceLms

        # cv ploting
        if cv_plot:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(black_img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            WindowName = "Image"
            cv2.imshow("Image", black_img)
            cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(10)

        if output_flm_video:
            # b_img = np.copy(black_img)
            # rgb_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB)
            video_writer.write(black_img.astype('uint8'))
        elif output_as_video:
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_writer.write(img)

        frame_id += 1

    cap.release()

    if output_as_video or output_flm_video:
        # Close the video writer
        video_writer.release()

    if output_flm_npy:
        flm_npy_path = os.path.join(output_path,
                                    "{}_flm".format(os.path.basename(os.path.normpath(output_path))))
        np.save(flm_npy_path, all_FLms)

    return all_FLms


def color_map_generator_rgb(all_FLms, ih=720, iw=1080):
    # image_rows, image_cols, _ = image.shape
    # brg_channel =
    # heatmap = np.zeros((total_frame, ih, iw))
    num_channel = 3
    red = 2
    total_frame = all_FLms.shape[0]
    f_heatmap = np.zeros((total_frame, ih, iw, num_channel))
    heatmap = np.zeros((ih, iw, num_channel))

    for f_idx in range(1, all_FLms.shape[0] - 1):
        prev = all_FLms[f_idx - 1]
        cur = all_FLms[f_idx]
        dist = np.linalg.norm(cur - prev, axis=1)

        for lm_idx in range(total_FLms):
            h, w = cur[lm_idx]
            # heatmap[f_idx, int(w), int(h)] = heatmap[f_idx - 1, int(w), int(h)] + dist[lm_idx]
            heatmap[int(w), int(h), red] = heatmap[int(w), int(h), red] + dist[lm_idx]

        img = heatmap
        WindowName = "Heatmap"
        cv2.imshow("Heatmap", img)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(30)
        f_heatmap[f_idx] = heatmap
    print("")


def color_map_generator(all_FLms, output_path, output_as_video=False):
    total_frame = all_FLms.shape[0]
    f_heatmap = np.zeros((total_frame, height, width))
    heatmap = np.zeros((height, width))

    if output_as_video:
        video_output = os.path.join(output_path,
                                    "{}_heatmap.avi".format(os.path.basename(os.path.normpath(output_path))))
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       camera_fps,
                                       (width, height))

    for f_idx in range(1, all_FLms.shape[0] - 1):
        prev = all_FLms[f_idx - 1]
        cur = all_FLms[f_idx]
        dist = np.linalg.norm(cur - prev, axis=1)

        for lm_idx in range(total_FLms):
            h, w = cur[lm_idx]
            # heatmap[f_idx, int(w), int(h)] = heatmap[f_idx - 1, int(w), int(h)] + dist[lm_idx]
            heatmap[int(w), int(h)] = heatmap[int(w), int(h)] + dist[lm_idx]

        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        WindowName = "Heatmap"
        cv2.imshow(WindowName, heatmapshow)
        if output_as_video:
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_writer.write(heatmapshow)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(20)

        # save to numpy
        f_heatmap[f_idx] = heatmap

    if output_as_video:
        # Close the video writer
        video_writer.release()

    return f_heatmap


#
# def color_map_generator2(all_FLms, ih=720, iw=1080):
#     # image_rows, image_cols, _ = image.shape
#     # brg_channel =
#     heatmap = np.zeros((total_frame, ih, iw))
#     # heatmap = np.zeros((ih, iw))
#
#     for f_idx in range(1, all_FLms.shape[0] - 1):
#         prev = all_FLms[f_idx - 1]
#         cur = all_FLms[f_idx]
#         dist = np.linalg.norm(cur - prev, axis=1)
#
#         for lm_idx in range(total_FLms):
#             h, w = cur[lm_idx]
#             heatmap[f_idx] = heatmap[f_idx-1]
#             heatmap[f_idx, int(w), int(h)] = heatmap[f_idx, int(w), int(h)] + dist[lm_idx]
#
#
#         # cv2.applyColorMap(heatmap[f_idx], cv2.COLORMAP_JET)
#
#     # normalize
#     # heatmap = heatmap / np.max(heatmap)
#
#     for f_idx in range(total_frame):
#         img = heatmap[f_idx]
#         WindowName = "Heatmap"
#         cv2.imshow("Heatmap", img)
#         cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
#         cv2.waitKey(30)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


if __name__ == "__main__":
    # video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Anger_1/Anger_1.avi"
    video_path = "C:/Users/Zber/Desktop/Subjects_Video/S2/Anger_1/Anger_1.avi"

    root_path = "C:/Users/Zber/Desktop/SavedData_MIMO/"
    # output_data_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/"
    # output_data_path = "C:/Users/Zber/Desktop/Subjects_Video/S1/"
    output_path = "C:/Users/Zber/Desktop/Subjects_Landmark/Test/"
    file_prefix = "Anger_1_flm"

    video_dir = "{}_{}"
    video_name = "{}_{}.avi"
    npy_name = "{}_{}"

    num_frame = 89
    save_npy = True

    _, keyparts = get_num_flm()

    # data
    score_data = np.zeros((num_frame, len(keyparts)))

    all_FLms = flm_detector(video_path, output_path, output_as_video=True, output_flm_video=False)
    key_score = distance(all_FLms)
    sum_score = np.sum(key_score, axis=1)

    # save npy file
    if save_npy:
        save_path = os.path.join(output_path, file_prefix)
        np.save(save_path, sum_score)
        print("Npy file saved")
