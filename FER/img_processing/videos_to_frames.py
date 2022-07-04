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


def detect_face_cv(img, padding_ratio=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/Zber/Documents/GitHub/face-alignment/emotion/haarcascade_frontalface_alt2.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=4, minSize=[200, 200], maxSize=[300, 300])
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
    reader1 = cv2.VideoCapture(video_filename)
    reader2 = cv2.VideoCapture(video_filename)

    # read first frame
    isok = False
    success = True
    # detect face cv
    while not isok and success:
        success, frame = reader1.read()
        if success:
            x, y, w, h, isok = detect_face_cv(frame)

    if not isok and not success:
        print("Cannot find face on {}".format(video_filename))
        # 30cm
        x, y, w, h = 636, 255, 253, 253
        # 70cm
        # x, y, w, h = 638, 292, 149, 149
        # 100cm
        # x, y, w, h = 640, 305, 113, 113
        # 150cm
        # x, y, w, h = 650, 309, 91, 91
        # 200cm
        # x, y, w, h = 657, 320, 66, 66
        # 250cm
        # x, y, w, h = 671, 280, 47, 47
        # 300cm
        # x, y, w, h = 660, 334, 40, 40
    # else:
    success, frame = reader2.read()
    count = 0
    while success:
        out_filepath = os.path.join(out_dir, file_template.format(count))
        frame = frame[y:y + h, x:x + w]
        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(out_filepath, frame)
        success, frame = reader2.read()
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

    # ['Standing_Jesse', 'Standing_Nancy', 'Ground_Jesse', 'Ground_Nancy', 'Distance_100cm']
    # ['Distance_70cm', 'Distance_100cm', 'Distance_150cm', 'Distance_200cm']
    # ['M1_0', 'M1_1','M2_0','M2_1','M3_0','M3_1', 'M1_2', 'M2_2', 'M3_2']
    # ['M1_0', 'M1_1','M2_0','M2_1','M3_0','M3_1', 'M1_2', 'M2_2', 'M3_2']
    # video_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Video\\Distance_300cm'

    # 70cm h=149, w=149, x=638, y=292

    # subs = ['30cm_60d']
    # subs = [ '30cm_90d']
    # subs = ['70cm_90d']
    # , '70cm_60d', '70cm_90d'

    #, '100cm_60d', '100cm_90d', '150cm_30d', '150cm_60d', '150cm_90d'
    subs = ['30cm_30d', '30cm_60d', '30cm_90d']
    # subs = ['200cm_30d', '200cm_60d', '200cm_90d', '250cm_30d', '250cm_60d', '250cm_90d']
    # subs = ['300cm_30d', '300cm_60d', '300cm_90d']
    for sub in subs:
        video_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Video\\{}'.format(sub)
        # the root output path where RGB frame folders should be created
        # rgb_out_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\Distance_300cm'
        rgb_out_path = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\{}'.format(sub)
        # the file extension that the videos have
        file_extension = '.avi'

        video_filenames = os.listdir(video_path)
        queue = Queue()
        [queue.put(video_filename) for video_filename in video_filenames]

        thread_job(queue, video_path, rgb_out_path, file_extension)

        NUM_THREADS = 12
        for i in range(NUM_THREADS):
            worker = threading.Thread(target=thread_job, args=(queue, video_path, rgb_out_path, file_extension))
            worker.start()

        print('waiting for all videos to be completed.', queue.qsize(), 'videos')
        print('This can take an hour or two depending on dataset size')
        queue.join()
        print('all done')
