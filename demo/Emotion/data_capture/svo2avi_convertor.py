import os
import cv2
import pyzed.sl as sl
import numpy as np
import sys

data_folder = 'C:\\Users\\Zber\\Desktop\\SavedData_MIMO'


# emotion_list = ['Anger']

# functions
def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def svo_to_npy(input_path, output_path, output_as_video=False, output_as_image=False, output_as_ndarray=False):
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
    right_image = sl.Mat()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    nb_frames = zed.get_svo_number_of_frames()

    left_ndarray = np.zeros((nb_frames, height, width, channel))
    right_ndarray = np.zeros_like(left_ndarray)

    if output_as_video:
        video_output = os.path.join(output_path, "{}.avi".format(os.path.basename(os.path.normpath(output_path))))
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
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            filename_left = os.path.join(output_path, ("left%s.png" % str(svo_position).zfill(6)))
            filename_right = os.path.join(output_path, ("right%s.png" % str(svo_position).zfill(6)))

            rgb_left = cv2.cvtColor(left_image.get_data(), cv2.COLOR_RGBA2RGB)
            rgb_right = cv2.cvtColor(right_image.get_data(), cv2.COLOR_RGBA2RGB)

            if output_as_video:
                video_writer.write(rgb_left)

            if output_as_image:
                # Save Left images
                cv2.imwrite(str(filename_left), rgb_left)
                # Save right images
                cv2.imwrite(str(filename_right), rgb_right)

                right_ndarray[svo_position] = rgb_right
                left_ndarray[svo_position] = rgb_left

            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        # Check if we have reached the end of the video
        if svo_position >= (nb_frames - 1):  # End of SVO
            sys.stdout.write("\n {} >>> Finish.\n".format(input_path))
            break
    # save npy file to output path
    if output_as_ndarray:
        np.savez_compressed(os.path.join(output_path, 'images'), left=left_ndarray, right=right_ndarray)

    if output_as_video:
        # Close the video writer
        video_writer.release()

    left_image.free(left_image.get_memory_type())
    right_image.free(right_image.get_memory_type())
    zed.close()


if __name__ == "__main__":

    start_index = 60
    end_index = 80

    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    for em in emotion_list:
        for i in range(start_index, end_index):
            svo_path = os.path.join(data_folder, "{}_{}.svo".format(em, i))
            output_folder = os.path.join(data_folder, "{}_{}".format(em, i))

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # process ZED svo file
            svo_to_npy(svo_path, output_folder, output_as_video=True)
