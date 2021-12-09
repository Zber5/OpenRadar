import os
import cv2
import threading
from queue import Queue

"""
Given individual video files (mp4, webm) on disk, creates a folder for
every video file and saves the video's RGB frames as jpeg files in that
folder.

It can be used to turn SomethingSomethingV2, which comes as 
many ".webm" files, into an RGB folder for each ".webm" file.
Uses multithreading to extract frames faster.

Modify the two filepaths at the bottom and then run this script.
"""


def video_to_rgb(video_filename, out_dir, resize_shape):
    file_template = 'frame_{0:012d}.jpg'
    reader = cv2.VideoCapture(video_filename)
    success, frame, = reader.read()  # read first frame

    count = 0
    while success:
        out_filepath = os.path.join(out_dir, file_template.format(count))
        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(out_filepath, frame)
        success, frame = reader.read()
        count += 1


def detect_face_cv(img, padding_ratio=1.1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/Zber/Documents/GitHub/face-alignment/emotion/haarcascade_frontalface_alt2.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=4)
    new_x = new_y = pad_w = pad_h = -1

    if len(faces) == 0:
        return (new_x, new_y, pad_w, pad_h, False)

    for (x, y, w, h) in faces:
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

        new_x = x - (pad_w - w) // 2
        new_y = y - (pad_h - h) // 2

        return (new_x, new_y, pad_w, pad_h, True)


def video_to_rgb_face(video_filename, out_dir, resize_shape):
    print(video_filename)
    file_template = 'frame_{0:012d}.jpg'
    reader = cv2.VideoCapture(video_filename)
    success, frame, = reader.read()  # read first frame

    # detect face cv

    x, y, w, h = detect_face_cv(frame)

    count = 0
    while success:
        out_filepath = os.path.join(out_dir, file_template.format(count))
        frame = frame[y:y + h, x:x + w]
        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(out_filepath, frame)
        success, frame = reader.read()
        count += 1

    print(" >> Finished!".format(video_filename))


def process_videofile(video_filename, video_path, rgb_out_path, file_extension: str = '.mp4'):
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extension, '')

    out_dir = os.path.join(rgb_out_path, video_filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # video_to_rgb(filepath, out_dir, resize_shape=(224, 224))
    video_to_rgb_face(filepath, out_dir, resize_shape=(224, 224))


def thread_job(queue, video_path, rgb_out_path, file_extension='.webm'):
    while not queue.empty():
        q = queue.get()
        video_filename = "{}{}".format(q, file_extension)
        vpath = os.path.join(video_path, q)
        process_videofile(video_filename, vpath, rgb_out_path, file_extension=file_extension)
        queue.task_done()


if __name__ == '__main__':
    # the path to the folder which contains all video files (mp4, webm, or other)
    video_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Video\\S0\\Anger_26'
    # the root output path where RGB frame folders should be created
    rgb_out_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\S0'
    # the file extension that the videos have
    file_extension = '.avi'


    video_filenames = os.listdir(video_path)
    queue = Queue()
    [queue.put(video_filename) for video_filename in video_filenames]

    thread_job(queue, video_path, rgb_out_path, file_extension)

    # NUM_THREADS = 16
    # for i in range(NUM_THREADS):
    #     worker = threading.Thread(target=thread_job, args=(queue, video_path, rgb_out_path, file_extension))
    #     worker.start()
    #
    # print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    # print('This can take an hour or two depending on dataset size')
    # queue.join()
    # print('all done')
