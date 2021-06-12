import pandas as pd
import numpy as np
from tsfresh import extract_features

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.metrics import classification_report
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, settings

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from copy import deepcopy


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
                raise AttributeError("The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X


if __name__ == '__main__':
    # df_ts, y = load_robot_execution_failures()
    # X = pd.DataFrame(index=y.index)
    class_names = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_8-10-5-3_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_7-9-6-4_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff1_segment_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_7-9-6-4_{}.npy'
    save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_8-10-5-3_{}.npy'

    # va_list = ['VA_{}'.format(i) for i in [8, 10, 7, 9, 6, 4, 5, 3]]
    va_list = ['VA_{}'.format(i) for i in [7, 9, 6, 4]]
    # va_list = ['VA_{}'.format(i) for i in [8, 10, 5, 3]]

    is_diff = True
    is_segment = True

    x = np.load(save_path.format('x'))
    y = np.load(save_path.format('y'))

    length = 150
    if is_diff:
        length = length -1

    if is_segment:
        length = 80

    num_data = 140
    num_rows = num_data * length
    num_colums = len(va_list) + 2

    data = np.zeros((num_rows, num_colums))
    label = np.zeros((num_data, 2))

    id = 0
    for xi, yi in zip(x, y):
        d = np.zeros((length, num_colums))
        d[:, 0] = id
        xi = xi.T
        d[:, 1] = np.arange(0, len(xi)).astype(int)
        d[:, 2:] = xi

        label[id] = np.asarray([id, yi])
        data[id * length:(id + 1) * length] = d
        id += 1

    # antenna_order = [7, 9, 6, 4] if device_v else [6, 7, 4, 9]
    # antenna_order = [8, 10, 5, 3] if device_v else [5, 8, 3, 10]

    # va_list = ['VA_{}'.format(i) for i in [8, 10, 7, 9, 6, 4, 5, 3]]
    # va_list = ['VA_{}'.format(i) for i in [7, 9, 6, 4]]
    # va_list = ['VA_{}'.format(i) for i in [8, 10, 5, 3]]

    df_x = pd.DataFrame(data, columns=['id', 'time'] + va_list)

    df_x = df_x.astype({'id': int, 'time': int})

    df_y = pd.DataFrame(label)

    X = pd.DataFrame(index=df_y.index)

    settings = ComprehensiveFCParameters()
    # settings = MinimalFCParameters()

    # settings = {
    #     "length": None,
    #     "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    # }

    extracted_features = extract_features(df_x, column_id="id", column_sort="time", impute_function=impute, default_fc_parameters=settings)
    # extracted_features = extract_features(df_x, column_id="id", column_sort="time", impute_function=impute)
    # print(extracted_features)

    features_filtered = select_features(extracted_features, y)
    print(features_filtered.head(1).to_string)

    X_train, X_test, y_train, y_test = train_test_split(features_filtered, y, random_state=0, test_size=0.2)

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
    classifier = svm.SVC(kernel='linear').fit(X_train_pca, y_train)

    np.set_printoptions(precision=2)

    # y_pred = classifier.predict(X_test)
    y_pred = classifier.predict(X_test_pca)

    print(classification_report(y_test, y_pred, target_names=class_names))

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        # disp = plot_confusion_matrix(classifier, X_test, y_test,
        #                              display_labels=class_names,
        #                              cmap=plt.cm.Blues,
        #                              normalize=normalize)

        disp = plot_confusion_matrix(classifier, X_test_pca, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

    # pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
    #                      ('classifier', RandomForestClassifier())])
    #
    # pipeline.set_params(augmenter__timeseries_container=df_x)
    # pipeline.fit(X, y)
    #
