import pandas as pd
import numpy as np
from tsfresh import extract_features

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd
from tsfresh import select_features
import tsfresh
from tsfresh.utilities.dataframe_functions import impute
from sklearn.metrics import classification_report
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, settings

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from copy import deepcopy
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score

antenna_order = [8, 10, 7, 9, 6, 4, 5, 3]
seed = 1234
np.random.seed(seed)


class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it
    compatible with pandas DataFrames.
    """

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError(
                    "The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X


def svd(sig):
    import matplotlib._color_data as mcd
    u, s, vh = np.linalg.svd(sig, full_matrices=False)
    # m = sig.shape[0]
    # n = sig.shape[1]
    #
    # k = 5
    #
    # # get Sigma/W
    # Sigma = np.zeros((m, n))
    # for i in range(m):
    #     Sigma[i, i] = s[i]
    #
    # # reconstruction
    # approx_sig = u @ Sigma[:, :k] @ vh[:k, :]
    #
    # # transformed Data
    # trans = u @ Sigma
    #
    # # 2D projection of transformed data
    # k1 = 2
    # trans_dd = trans[:, :k1]
    # trans_x = trans_dd[:, 0]
    # trans_y = trans_dd[:, 1]
    #
    # # colors
    # tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
    #
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    #
    # ant_index = 0
    # for o, color in zip(antenna_order, tab_color):
    #     axes[0][0].plot(sig[ant_index], c=color, label="VA_{}".format(o), lw=2)
    #     axes[0][1].plot(trans[ant_index], c=color, lw=2)
    #     axes[1][0].plot(approx_sig[ant_index], c=color, lw=2)
    #     axes[1][1].scatter(trans_x[ant_index], trans_y[ant_index], c=color, s=15)
    #     ant_index += 1
    # handles, labels = axes[0][0].get_legend_handles_labels()
    # plt.tight_layout()
    # plt.legend(handles, labels, bbox_to_anchor=(0.1, 2.13), ncol=8, loc='upper center', borderaxespad=0., fontsize=10)
    # plt.show()

    return u, s, vh


def select_channel(x, select_va):
    m = len(select_va)
    xs = x.shape
    select_x = np.zeros((xs[0], m, xs[2]))

    for i, va in enumerate(select_va):
        p = antenna_order.index(va)
        select_x[:, i, :] = x[:, p, :]

    return select_x


def select_data(label=[0, 1, 2]):
    return 0


def twin_sliding_window(imag_length, sensor_length, imag_window=6, sensor_window=20, imag_step=3, sensor_step=10):
    imag_start = 0
    sensor_start = 0

    while (imag_start + imag_window) <= imag_length and (sensor_start + sensor_window) <= sensor_length:
        yield (imag_start, imag_start + imag_window), (sensor_start, sensor_start + sensor_window)
        imag_start += imag_step
        sensor_start += sensor_step


def format_dataset(sensor, imag, label, imag_window=6, sensor_window=20, imag_step=3, sensor_step=10):
    num_data = np.shape(sensor)[0]
    sensor_channels = np.shape(sensor)[1]
    imag_channels = np.shape(imag)[1]
    sensor_length = np.shape(sensor)[2]
    imag_length = np.shape(imag)[2]

    num_segment = imag_length // imag_step

    x = np.zeros((num_segment * num_data, sensor_channels, sensor_window))
    y = np.zeros((num_segment * num_data, imag_channels))
    l = np.zeros((num_segment * num_data))
    index = 0
    for i in range(num_data):
        label_content = label[i]
        for imag_idx, sensor_idx in twin_sliding_window(imag_length, sensor_length, imag_window, sensor_window,
                                                        imag_step, sensor_step):
            sensor_start, sensor_end = sensor_idx
            imag_start, imag_end = imag_idx
            x[index] = sensor[i, :, sensor_start:sensor_end]
            imag_mean = np.mean(imag[i, :, imag_start:imag_end], axis=1)
            y[index] = imag_mean
            l[index] = label_content
            index += 1

    return x, y, l


if __name__ == '__main__':
    # df_ts, y = load_robot_execution_failures()
    # X = pd.DataFrame(index=y.index)
    # class_names = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    class_names = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']

    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_8-10-5-3_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_7-9-6-4_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_{}.npy'

    # segemented data
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff1_segment_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_7-9-6-4_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_8-10-5-3_{}.npy'
    save_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/ml_{}_b8_c5_{}.npy"

    # manually select
    # va_list = ['VA_{}'.format(i) for i in [8, 10, 7, 9, 6, 4, 5, 3]]
    # va_list = ['VA_{}'.format(i) for i in [7, 9, 6, 4]]
    # va_list = ['VA_{}'.format(i) for i in [8, 10, 5, 3]]
    va_list = ['VA_{}'.format(i) for i in range(12)]

    is_diff = True
    is_segment = True

    x = np.load(save_path.format('sensor', 'x'))
    y = np.load(save_path.format('image', 'y'))

    # svd
    # y[41]

    # select_va = [5, 3, 7, 4, 6]
    # select_va = [3, 7, 6, 9, 4]
    # va_list = ['VA_{}'.format(i) for i in select_va]

    # select_x
    # x = select_channel(x, select_va)

    length = 300
    if is_diff:
        length = length - 1

    if is_segment:
        length = 30

    num_data = np.shape(x)[0]
    num_rows = num_data * length
    num_colums = len(va_list) + 2

    # svd feature calculation
    # svd_feature = np.zeros((num_data, len(va_list)))
    # for index, ix in enumerate(x):
    #     u, s, vh = svd(ix)
    #     svd_feature[index] = s
    #
    # svd_feature = pd.DataFrame(svd_feature, columns=['svd_{}'.format(i) for i in range(8)])

    data = np.zeros((num_rows, num_colums))
    label = np.zeros((num_data, len(class_names) + 1))

    id = 0
    for xi, yi in zip(x, y):
        d = np.zeros((length, num_colums))
        d[:, 0] = id
        xi = xi.T
        d[:, 1] = np.arange(0, len(xi)).astype(int)
        d[:, 2:] = xi

        label[id, 0] = id
        label[id, 1:] = yi
        data[id * length:(id + 1) * length] = d
        id += 1

    # antenna_order = [7, 9, 6, 4] if device_v else [6, 7, 4, 9]
    # antenna_order = [8, 10, 5, 3] if device_v else [5, 8, 3, 10]

    # va_list = ['VA_{}'.format(i) for i in [8, 10, 7, 9, 6, 4, 5, 3]]
    # va_list = ['VA_{}'.format(i) for i in [7, 9, 6, 4]]
    # va_list = ['VA_{}'.format(i) for i in [8, 10, 5, 3]]

    df_x = pd.DataFrame(data, columns=['id', 'time'] + va_list)

    df_x = df_x.astype({'id': int, 'time': int})

    df_y = pd.DataFrame(label, columns=['id'] + class_names)

    df_data = pd.DataFrame(index=df_y.index)

    # lb = preprocessing.MultiLabelBinarizer()

    # feature selection
    # settings = ComprehensiveFCParameters()
    settings = MinimalFCParameters()

    # settings = {
    #     "length": None,
    #     "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    # }

    extracted_features = extract_features(df_x, column_id="id", column_sort="time", impute_function=impute,
                                          default_fc_parameters=settings)

    # extracted_features = pd.merge(extracted_features, svd_feature, left_index=True, right_index=True)

    # extracted_features = extract_features(df_x, column_id="id", column_sort="time", impute_function=impute)
    # print(extracted_features)

    # features_filtered = select_features(extracted_features, df_y, ml_task='classification', multiclass=True,
    #                                     n_significant=3)
    #
    # features_filtered = select_features(extracted_features, y)

    # we can easily construct the corresponding settings object
    # kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features_filtered)

    #

    # save feature dictionary
    # import pickle
    # a_file = open("setting/feature_dic.pkl", "wb")
    # pickle.dump(kind_to_fc_parameters, a_file)
    # a_file.close()

    # load
    # a_file = open("setting/feature_dic.pkl", "rb")
    # output = pickle.load(a_file)
    # print(output)
    # a_file.close()

    # import json
    # a_file = open("setting/feature_dic.json", "w")
    # json.dump(kind_to_fc_parameters, a_file, indent=4)
    # a_file.close()

    # features_filtered = pd.merge(features_filtered, svd_feature, left_index=True, right_index=True)

    # print(features_filtered.head(1).to_string)

    # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(extracted_features, df_y, random_state=0, test_size=0.2)

    # X_train = pd.DataFrame(data=X_train)

    # PCA train
    pca_train = PCAForPandas(n_components=20)
    X_train_pca = pca_train.fit_transform(X_train)
    # X_train_pca.index += 1

    print(X_train_pca.tail())

    # PCA test
    X_test_1 = deepcopy(X_test)
    X_test_pca = pca_train.transform(X_test_1)

    # reset index to keep original index from robot example
    # X_test_pca.index = [87, 88]

    print(X_test_pca.tail())

    # classifier = svm.SVC(kernel='linear').fit(X_train, y_train)
    # classifier = svm.SVC(kernel='linear').fit(X_train_pca, y_train)

    SVC_pipeline = Pipeline([('classifier', OneVsRestClassifier(LinearSVC(max_iter=2000), n_jobs=8))])
    # NB_pipeline = Pipeline([('classifier', OneVsRestClassifier(MultinomialNB(
    #                         fit_prior=True, class_prior=None), n_jobs=8))])

    class_names = ['AU01', 'AU02', 'AU06', 'AU07', 'AU10', 'AU14', 'AU15']

    for category in class_names:
        print("{}: {}".format(category, int(np.sum(y_test[category]))))

    for category in class_names:
        print('... Processing {}'.format(category))
        # train the model using X_dtm & y
        SVC_pipeline.fit(X_train_pca, y_train[category])
        # compute the testing accuracy
        prediction = SVC_pipeline.predict(X_test_pca)
        acc = accuracy_score(y_test[category], prediction)
        f1 = f1_score(y_test[category], prediction, average='weighted')
        print('{} >> ACC: {:.2%}, F1: {:.2%}'.format(category, acc, f1))

    # np.set_printoptions(precision=2)
    #
    # # y_pred = classifier.predict(X_test)
    # y_pred = classifier.predict(X_test_pca)
    #
    # print(classification_report(y_test, y_pred, target_names=class_names))
    #
    # # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #                   ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     # disp = plot_confusion_matrix(classifier, X_test, y_test,
    #     #                              display_labels=class_names,
    #     #                              cmap=plt.cm.Blues,
    #     #                              normalize=normalize)
    #
    #     disp = plot_confusion_matrix(classifier, X_test_pca, y_test,
    #                                  display_labels=class_names,
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp.ax_.set_title(title)
    #
    #     print(title)
    #     print(disp.confusion_matrix)
    #
    # plt.show()
