import subprocess
import os
import threading
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing&surprise_0.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing2ground_0.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/1m_sit_0.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/stand_1m_move_1.avi"
    input_loc = "C:/Users/Zber/Desktop/Subjects/Test/sit_1m_3.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/1m_ground_0.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/ground_1m_1.avi"
    # input_loc = "C:/Users/Zber/Desktop/Subjects/Test/Standing_1.avi"

    save_folder = "C:/Users/Zber/Desktop/Subjects/Test/image_angle_sit_1m_3"

    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    print("Converting video../n")
    count = 0

    angle_data = np.zeros((video_length))

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            'C:/Users/Zber/Documents/GitHub/face-alignment/emotion/haarcascade_frontalface_alt2.xml')

        faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=4, minSize=[80, 80])
        new_x = new_y = pad_w = pad_h = -1

        if len(faces) == 0:
            angle_data[count] = 0
            count = count + 1
            continue

        for (x, y, w, h) in faces:
            f = frame[y:y + h, x:x + w]

            cv2.imshow("landmark image", f)
            cv2.waitKey(50)

            ang = y + (h // 2)
            ang = ang / 720 * 70
            angle_data[count] = ang
            # angle_data[count] = np.abs(70 - ang) * 7 / 12

        count = count + 1

        # If there are no more frames left
        if (count > (video_length - 1)):
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames./n%d frames extracted" % count)
            break


    plt.plot(angle_data)
    plt.show()

    # np.save("C:/Users/Zber/Desktop/Subjects/Test/image_angle_standing&surprise", angle_data)
    # np.save("C:/Users/Zber/Desktop/Subjects/Test/image_angle_sit_1m_move", angle_data)
    # np.save("C:/Users/Zber/Desktop/Subjects/Test/image_angle_ground_1m_1", angle_data)
    # np.save("C:/Users/Zber/Desktop/Subjects/Test/image_angle_standing_1", angle_data)
    np.save(save_folder, angle_data)

    print("")
