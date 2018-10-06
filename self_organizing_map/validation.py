from self_organizing_map.objects import SOM, RawDataConverter
from self_organizing_map.utility_funcs import mat_compare_v4
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # define path to folder that contains labeled samples
    folder_path = r'C:\Users\Apex\PycharmProjects\sample_data_clone\data\labeled_samples_sensorII_1600Hz_channel1'

    # initiate raw data converter
    rdc = RawDataConverter(path=folder_path, axis='y', NFFT=1024)

    # generate train and test data
    X_train, X_test, y_train, y_test = rdc.train_test_gen([13], [5])
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    #X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
    training_data = []
    for i in np.unique(y_train):
        row_idx = np.where(y_train == i)[0]
        debug = X_train[row_idx, :]
        med_debug = np.median(debug, axis=0, keepdims=False).reshape(1, -1)
        training_data.append(med_debug)
    training_data = np.vstack(training_data)
    # stats = StatFeatureExtractor(X_train).features
    # X_test = StatFeatureExtractor(X_test).features
    # instantiate SOM
    som = SOM(20, 40, X_train.shape[1], 400, 0.3, wanted_clusters=10, metric='manhattan')
    # train
    som.fit(X_train[::10])

    # plot U-matrix
    umatrix, axis1 = plt.subplots(1, 1)
    axis1.imshow(som.get_umatrix(), cmap='magma')
    plt.show()

    # predict
    y_pred = som.predict_w_umatrix(X_test)
    som.animate_bmu_trajectory()

    # adjust labeling
    y_pred = mat_compare_v4(y_test, y_pred)

    comparison, axis2 = plt.subplots(1, 1)
    axis2.plot(y_pred, c='r')
    axis2.plot(y_test, c='k')
    plt.show()


if __name__ == '__main__':
    main()
