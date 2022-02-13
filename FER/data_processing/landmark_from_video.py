import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
import mediapipe as mp
import cv2
from imutils.video import count_frames
from FER.utils import parseConfigFile, arange_tx, get_label
from queue import Queue
import threading
from mmwave.dsp.utils import Window

from PIL import Image
from numpy import asarray

total_FLms = 468


def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def load_image_as_ndarray(image_path):
    image = Image.open(image_path)
    # convert image to numpy array
    img_array = asarray(image)
    return img_array


def save_landmark_data(video_path, save_path, subject_id):
    total_frames = 89

    # npy path
    folder_name = os.path.basename(video_path).replace(".avi", "")
    landmark_folder = os.path.join(save_path, subject_id)
    npy_path = os.path.join(landmark_folder, folder_name)
    makefile(landmark_folder)

    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

    cap = cv2.VideoCapture(video_path)

    counted_frames = count_frames(video_path)

    if counted_frames < total_frames:
        print("{} number of frames is less than 89".format(counted_frames))
    #     return 0

    np_faceLms = np.zeros((total_frames, total_FLms, 3))

    frame_id = 0
    flag = False
    while True:
        if frame_id >= total_frames:
            break
        if frame_id >= counted_frames:
            np_faceLms[frame_id] = np_faceLms[frame_id - 1]

        else:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not success:
                break
            results = faceMesh.process(imgRGB)

            if results.multi_face_landmarks:
                faceLms = results.multi_face_landmarks[0]

                for id, lm in enumerate(faceLms.landmark):
                    np_faceLms[frame_id, id] = [lm.x, lm.y, lm.z]
            else:
                flag = True
                # print("Face landmark in {} - {} has not been detected <<>>".format(npy_path, frame_id))
        frame_id += 1

    if flag:
        print("Face landmark in {} has not been detected <<>>".format(npy_path))
    else:
        # save npy file
        np.save(npy_path, np_faceLms)
        print("{} npy file has saved!".format(npy_path))


def thread_job(queue):
    while not queue.empty():
        frame_folder_path, subid = queue.get()

        save_landmark_data(frame_folder_path, save_path, subid)
        queue.task_done()


def check_file_existence(emo, id, sub, save_path):
    sub_folder = os.path.join(save_path, sub)
    if not os.path.exists(sub_folder):
        return False
    npy_file_name = "{}_{}.npy".format(emo, id)

    exists_files = os.listdir(sub_folder)
    if npy_file_name in exists_files:
        return True
    else:
        return False


if __name__ == '__main__':

    # config paths
    save_path = "C:/Users/Zber/Desktop/Subjects_Landmark_video"
    root_path = "C:/Users/Zber/Desktop/Subjects_Video"
    data_path = "{}_{}"
    video_path = "{}_{}.avi"

    # start index
    subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    queue = Queue()

    start_index = 0
    end_index = 30

    for sub in subs:
        for l, e in enumerate(emotion_list):
            for i in range(start_index, end_index):
                if not check_file_existence(e, i, sub, save_path):
                    frame_folder_path = os.path.join(root_path, sub, data_path.format(e, i), video_path.format(e, i))
                    queue.put([frame_folder_path, sub])

    thread_job(queue)

    # NUM_THREADS = 16
    # for i in range(NUM_THREADS):
    #     worker = threading.Thread(target=thread_job, args=(queue,))
    #     worker.start()
    #
    # print('waiting for all tasks to be completed.', queue.qsize(), 'tasks')
    # print('This can take an hour or two depending on dataset size')
    # queue.join()
    # print('all done')
