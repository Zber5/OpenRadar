import os
import cv2
import pyzed.sl as sl
import numpy as np
import sys
from queue import Queue
import threading


# functions
def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def svo_to_npy(input_path, output_path, output_as_video=False, output_as_image=False, output_as_ndarray=False):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    channel = 3
    # width_sbs = width * 2

    # Prepare single image containers
    left_image = sl.Mat()
    # right_image = sl.Mat()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    nb_frames = 89

    left_ndarray = np.zeros((nb_frames, height, width, channel))
    # right_ndarray = np.zeros_like(left_ndarray)

    if output_as_video:
        # left_image = sl.Mat()
        # prefix = os.path.basename(os.path.normpath(input_path))[:-4].replace('_2_', '_')
        prefix = os.path.basename(os.path.normpath(input_path))[:-4]
        video_output = os.path.join(output_path, "{}.avi".format(prefix))
        # video_output = os.path.join(output_path, "{}.avi".format(os.path.basename(os.path.normpath(input_path))[:-4]))
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       zed.get_camera_information().camera_fps,
                                       (width, height))
        if not video_writer.isOpened():
            sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
            zed.close()
            exit()

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            # Retrieve SVO images
            # zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            filename_left = os.path.join(output_path, ("left%s.png" % str(svo_position).zfill(6)))
            # filename_right = os.path.join(output_path, ("right%s.png" % str(svo_position).zfill(6)))

            rgb_left = cv2.cvtColor(left_image.get_data(), cv2.COLOR_RGBA2RGB)
            # rgb_right = cv2.cvtColor(right_image.get_data(), cv2.COLOR_RGBA2RGB)

            if output_as_video:
                video_writer.write(rgb_left)

            if output_as_image:
                # Save Left images
                cv2.imwrite(str(filename_left), rgb_left)
                # Save right images
                # cv2.imwrite(str(filename_right), rgb_right)

                # right_ndarray[svo_position] = rgb_right
                left_ndarray[svo_position] = rgb_left

            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        # Check if we have reached the end of the video
        if svo_position >= (nb_frames - 1):  # End of SVO
            sys.stdout.write("\n {} >>> Finish.\n".format(input_path))
            break
    # save npy file to output path
    if output_as_ndarray:
        # np.savez_compressed(os.path.join(output_path, 'images'), left=left_ndarray, right=right_ndarray)
        print("")
    if output_as_video:
        # Close the video writer
        video_writer.release()

    left_image.free(left_image.get_memory_type())
    # right_image.free(right_image.get_memory_type())
    zed.close()


def get_folder_size(path):
    size = 0
    for ele in os.scandir(path):
        size += os.path.getsize(ele)
    return size


def thread_job(queue, output_folder):
    while not queue.empty():
        video_path = queue.get()
        sub = os.path.basename(os.path.dirname(video_path))
        folder = os.path.basename(os.path.normpath(video_path))[:-4]
        output_path = os.path.join(output_folder, sub, folder)
        svo_to_npy(video_path, output_path, output_as_video=True)
        queue.task_done()


if __name__ == "__main__":

    # ========= Single file convert =========
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing&surprise_0.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_sit_0.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_ground_0.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/ground_1m_1.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/Standing_1.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/sit_1m_move_1.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/sit_1m_3.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Distance_1m_stand/Joy_0.svo"
    # video_path = "C:/Users/Zber/Desktop/Subjects/Test/stand_1m_head_move_0.svo"
    # output_path = "C:/Users/Zber/Desktop/Subjects/Test/"
    # svo_to_npy(video_path, output_path, output_as_video=True)

    # ========= Multi file convert =========
    all = []
    start_index = 0
    end_index = 10
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    for em in emotion_list:
        for i in range(start_index, end_index):
            all.append("{}_{}".format(em, i))

    all = np.asarray(all)

    # subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    # subs = ['S8', 'S9']
    # subs = ['Standing_Jesse', 'Standing_Nancy', 'Ground_Jesse', 'Ground_Nancy', 'Distance_100cm']
    # subs = ['Standing_Jesse', 'Ground_Jesse']
    # subs = ['Distance_100cm_Nancy']
    # subs = ['Distance_70cm', 'Distance_100cm', 'Distance_150cm', 'Distance_200cm', 'Distance_250cm']
    # subs = ['Distance_300cm']
    # subs = ['Camera']
    # subs = ['30cm_30d', '30cm_60d', '30cm_90d', '70cm_30d', '70cm_60d', '70cm_90d']
    # subs = ['100cm_30d', '100cm_60d', '100cm_90d', '150cm_30d', '150cm_60d', '150cm_90d']
    # subs = ['200cm_30d', '200cm_60d', '200cm_90d', '250cm_30d', '250cm_60d', '250cm_90d']
    subs = ['300cm_30d', '300cm_60d', '300cm_90d']
    # subs = ['M1_2', 'M2_2', 'M3_2']
    size = 800000

    data_folder = 'D:\\Subjects\\'
    output_folder = 'C:\\Users\\Zber\\Desktop\\Subjects_Video'

    queue = Queue()

    for sub in subs:
        sub_folder = os.path.join(data_folder, sub)
        sub_output = os.path.join(output_folder, sub)
        content = [name for name in os.listdir(sub_output) if
                   os.path.isdir(os.path.join(sub_output, name)) and get_folder_size(
                       os.path.join(sub_output, name)) > size]
        tasks = all[~np.isin(all, content)]

        tasks_path = [os.path.join(sub_folder, "{}.svo".format(t)) for t in tasks]

        for tp in tasks_path:
            queue.put(tp)

    thread_job(queue, output_folder)

    # multi threadings
    # NUM_THREADS = 4
    # for i in range(NUM_THREADS):
    #     worker = threading.Thread(target=thread_job, args=(queue, output_folder))
    #     worker.start()
    #
    # print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    # print('This can take an hour or two depending on dataset size')
    # queue.join()
    # print('all done')
