import os

def rename(old_name):
    old_path = os.path.join(data_folder, old_name)
    new_path = old_path.replace('_2_', '_')
    os.rename(old_path, new_path)
    print("Rename from {} to {}".format(old_path, new_path))


if __name__ == "__main__":

    start_index = 30
    end_index = 40
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    data_folder = 'D:\\Subjects\\S2'

    # rename files
    for emo in emotion_list:
        for idx in range(start_index, end_index):

            # svo
            svo_name = "{}_2_{}.svo".format(emo, idx)
            rename(svo_name)

            # bin
            bin_name = "{}_2_{}_Raw_0.bin".format(emo, idx)
            rename(bin_name)

            # log
            log_name = "{}_2_{}_Raw_LogFile.csv".format(emo, idx)
            rename(log_name)
