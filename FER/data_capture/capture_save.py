# ZED Stereo camera control
import cv2
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

import serial
import time
import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
import os
import json
# import keyboard

# Change the configuration file name
# configFileName = 'C:/Users/Zber/Desktop/Tx3_bestRangeResolution.cfg'
# configFileName = 'C:/Users/Zber/Desktop/bestRangeResolution.cfg'


# 10 fps,
# configFileName = 'C:/Users/Zber/Desktop/vod_vs_18xx_100fps_8rr.cfg'

# 100 fps, 1 chirp, 20 adc sample
# configFileName = 'C:/Users/Zber/Desktop/vod_vs_18xx_100fps.cfg'

# 100 fps vocal print config
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/vocal_print_config.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx2_rx4_RangeResolution0.8cm.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_modify.cfg'
# configFileName = 'C:/Users/Zber/Desktop/profiles/profile_advanced_subframe.cfg'
# configFileName = 'C:/Users/Zber/Desktop/profiles/profile_3d_aop.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_modify2.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_modify2_realtime.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_modify3-velocity.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/100fps_estimator.cfg'

# AOP 3s
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_5s.cfg'
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_30s_20fps.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_20s_50fps_2.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s_10fps.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s_50fps.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/radarProfile.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/Front.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/xwr18xx_profile_front.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_10s.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx1_rx4_2.cfg'

# beamforming
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_beamforming_3region.cfg'

# datacard config
datacard_config = 'C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/PostProc/datacard_config.json'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0
real_time = False


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial('COM3', 115200)
    Dataport = serial.Serial('COM4', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.001)

    # CLIport.close()
    # Dataport.close()

    return CLIport, Dataport


def send_stop():
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial('COM3', 115200)
    Dataport = serial.Serial('COM4', 921600)

    CLIport.write(('sensorStop' + '\n').encode())
    print("Sensor Stopped!")

    CLIport.close()
    Dataport.close()


# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName, numTxAnt=3):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        # numRxAnt = numRxAntennas
        # numTxAnt = numTxAntennas

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            adcStartTime = float(splitWords[4])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(splitWords[11])

            configParameters['startFreq'] = startFreq
            configParameters['idleTime'] = idleTime
            configParameters['adcStartTime'] = adcStartTime
            configParameters['rampEndTime'] = rampEndTime
            configParameters['numAdcSamples'] = numAdcSamples
            configParameters['digOutSampleRate'] = digOutSampleRate
            configParameters['freqSlopeConst'] = freqSlopeConst

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

            configParameters['numLoops'] = numLoops
            configParameters['numFrames'] = numFrames
            configParameters['framePeriodicity'] = framePeriodicity

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
            2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


# ------------------------------------------------------------------

# Funtion to read and parse the incoming data

# def readAndParseData18xx(Dataport, configParameters):
#     global byteBuffer, byteBufferLength
#
#     # Constants
#     OBJ_STRUCT_SIZE_BYTES = 12;
#     BYTE_VEC_ACC_MAX_SIZE = 2 ** 15;
#     MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
#     MMWDEMO_UART_MSG_RANGE_PROFILE = 2;
#     maxBufferSize = 2 ** 15;
#     tlvHeaderLengthInBytes = 8;
#     pointLengthInBytes = 16;
#     magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
#
#     # Initialize variables
#     magicOK = 0  # Checks if magic number has been read
#     dataOK = 0  # Checks if the data has been read correctly
#     frameNumber = 0
#     detObj = {}
#
#     readBuffer = Dataport.read(Dataport.in_waiting)
#     byteVec = np.frombuffer(readBuffer, dtype='uint8')
#     byteCount = len(byteVec)
#
#     # Check that the buffer is not full, and then add the data to the buffer
#     if (byteBufferLength + byteCount) < maxBufferSize:
#         byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
#         byteBufferLength = byteBufferLength + byteCount
#
#     # Check that the buffer has some data
#     if byteBufferLength > 16:
#
#         # Check for all possible locations of the magic word
#         possibleLocs = np.where(byteBuffer == magicWord[0])[0]
#
#         # Confirm that is the beginning of the magic word and store the index in startIdx
#         startIdx = []
#         for loc in possibleLocs:
#             check = byteBuffer[loc:loc + 8]
#             if np.all(check == magicWord):
#                 startIdx.append(loc)
#
#         # Check that startIdx is not empty
#         if startIdx:
#
#             # Remove the data before the first start index
#             if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
#                 byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
#                 byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
#                                                                        dtype='uint8')
#                 byteBufferLength = byteBufferLength - startIdx[0]
#
#             # Check that there have no errors with the byte buffer length
#             if byteBufferLength < 0:
#                 byteBufferLength = 0
#
#             # word array to convert 4 bytes to a 32 bit number
#             word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
#
#             # Read the total packet length
#             totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)
#
#             # Check that all the packet has been read
#             if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
#                 magicOK = 1
#
#     # If magicOK is equal to 1 then process the message
#     if magicOK:
#         # word array to convert 4 bytes to a 32 bit number
#         word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
#
#         # Initialize the pointer index
#         idX = 0
#
#         # Read the header
#         magicNumber = byteBuffer[idX:idX + 8]
#         idX += 8
#         version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
#         idX += 4
#         totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#         platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
#         idX += 4
#         frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#         timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#         numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#         numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#         subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
#         idX += 4
#
#         # Read the TLV messages
#         for tlvIdx in range(numTLVs):
#
#             # word array to convert 4 bytes to a 32 bit number
#             word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
#
#             # Check the header of the TLV message
#             tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
#             idX += 4
#             tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
#             idX += 4
#
#             # Read the data depending on the TLV message
#             if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
#
#                 # Initialize the arrays
#                 x = np.zeros(numDetectedObj, dtype=np.float32)
#                 y = np.zeros(numDetectedObj, dtype=np.float32)
#                 z = np.zeros(numDetectedObj, dtype=np.float32)
#                 velocity = np.zeros(numDetectedObj, dtype=np.float32)
#
#                 for objectNum in range(numDetectedObj):
#                     # Read the data for each object
#                     x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
#                     idX += 4
#                     y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
#                     idX += 4
#                     z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
#                     idX += 4
#                     velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
#                     idX += 4
#
#                 # Store the data in the detObj dictionary
#                 detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity}
#                 dataOK = 1
#
#         # Remove already processed data
#         if idX > 0 and byteBufferLength > idX:
#             shiftSize = totalPacketLen
#
#             byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
#             byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
#                                                                  dtype='uint8')
#             byteBufferLength = byteBufferLength - shiftSize
#
#             # Check that there are no errors with the buffer length
#             if byteBufferLength < 0:
#                 byteBufferLength = 0
#
#     return dataOK, frameNumber, detObj


# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
# def update():
#     dataOk = 0
#     global detObj
#     x = []
#     y = []
#
#     # Read and parse the received data
#     dataOk, frameNumber, detObj = readAndParseData18xx(Dataport, configParameters)
#
#     if dataOk and len(detObj["x"]) > 0:
#         # print(detObj)
#         x = -detObj["x"]
#         y = detObj["y"]
#
#         # s.setData(x, y)
#         # QtGui.QApplication.processEvents()
#
#     return dataOk


def update_data_config(jsonfile, file_prefix, file_basepath):
    with open(jsonfile, 'r+') as f:
        data = json.load(f)
        data['DCA1000Config']['captureConfig']['fileBasePath'] = file_basepath
        data['DCA1000Config']['captureConfig']['filePrefix'] = file_prefix

        f.seek(0)  # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()  # remove remaining part


# Function to
def send_cli():
    # import subprocess
    postproc_path = "C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/PostProc"
    json_file_name = "datacard_config.json"
    cwd = os.getcwd()
    os.chdir(postproc_path)
    # os.system(f"DCA1000EVM_CLI_Control.exe reset_fpga {json_file_name}")
    os.system(f"DCA1000EVM_CLI_Control.exe fpga {json_file_name}")
    os.system(f"DCA1000EVM_CLI_Control.exe record {json_file_name}")
    os.system(f"DCA1000EVM_CLI_Control.exe start_record {json_file_name}")
    CLIport, Dataport = serialConfig(configFileName)

    # real time, terminate manually
    # if real_time:
    #     print("real-time data collecting, press Esc to exit: ")
    #     while True:
    #         try:
    #             if keyboard.is_pressed('Esc'):
    #                 print("you pressed Esc, quit data collection..")
    #                 os.system(f"DCA1000EVM_CLI_Control.exe stop_record {json_file_name}")
    #                 os.chdir(cwd)
    #                 time.sleep(1)
    #                 send_stop()
    #                 break
    #         except:
    #             break
    # os.system(f"DCA1000EVM_CLI_Control.exe stop_record {json_file_name}")
    # os.chdir(cwd)

    return CLIport, Dataport


# Get the configuration parameters from the configuration file
# configParameters = parseConfigFile(configFileName)


# -------------------------    MAIN   -----------------------------------------
if __name__ == "__main__":
    el = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Test"

    # Subjects
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S0"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S1"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S2"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S3"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S4"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S5"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S6"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\S7"
    ## file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\W1"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Test"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m_stand"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m_ground"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Multi_People_3"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Multi_People"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m_Nancy"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_70cm"
    # file_basepath = "D:\\Subjects\\Distance_150cm"
    # file_basepath = "D:\\Subjects\\Standing_Jesse"
    # file_basepath = "D:\\Subjects\\Ground_Nancy"
    # file_basepath = "D:\\Subjects\\W3"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_200cm"
    # file_basepath = "D:\\Subjects\\Distance_300cm"
    # file_basepath = "C:\\Users\\Zber\\Desktop\\Subjects\\Multi_People"

    file_basepath = "D:\\Subjects\\300cm_90d"
    file_basepath = "D:\\mmFer_Data\\Subjects\\Multi_People"

    # prefix = "200cm_Stand_Surprise{}"
    # prefix = "Neutral_{}"
    prefix = "100cm_stand_ground_{}"

    config = parseConfigFile(configFileName)

    record_time = config['numFrames'] / (1000 / config['framePeriodicity'])

    interval = 5
    start_record_index = 2
    end_record_index = 3

    # preparing camera
    init = sl.InitParameters()
    cam = sl.Camera()
    init.camera_fps = 30
    # if not cam.is_opened():
    #     print("Opening ZED Camera...")

    # if status != sl.ERROR_CODE.SUCCESS:
    #     print(repr(status))
    #     exit()
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    # print_camera_information(cam)

    WindowName = "ZED Camera"
    view_window = cv2.namedWindow(WindowName, cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow(WindowName, 300, 150)
    # cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)

    for i in range(start_record_index, end_record_index):
        # time_start = time.time()
        # preparing data config json file
        file_prefix = prefix.format(i)
        update_data_config(datacard_config, file_prefix, file_basepath)
        cv2.waitKey(20)

        # camera record setting
        video_filepath = os.path.join(file_basepath, file_prefix + '.svo')
        record_param = sl.RecordingParameters(video_filepath)

        # preparing mmwave sensor
        CLIport, Dataport = send_cli()
        status = cam.open(init)

        # sensor start
        CLIport.write(('sensorStart' + '\n').encode())
        # cv2.waitKey(5)
        sensor_start = time.time()
        # camera recording
        cam.enable_recording(record_param)

        video_start = time.time()
        while time.time() - video_start <= record_time:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat)
                cv2.imshow(WindowName, mat.get_data())
                cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(5)

        # sensor stop
        # CLIport.write(('sensorStop\n').encode())
        print("Send sensor stop command")

        # camera stop
        cam.disable_recording()
        cv2.destroyAllWindows()

        time.sleep(interval)

        CLIport.close()
        Dataport.close()

        # close camera
        mat.free(mat.get_memory_type())
        cam.close()

        print("Sensor starting time: {}".format(sensor_start))
        print("Video starting time: {}".format(video_start))
