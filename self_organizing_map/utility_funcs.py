import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from skimage import filters


def get_watershed(image, save_labels=False, plot=False, wanted_clusters=None):
    """
    uses watershed transformation to get clusters from the presented U-matrix of the SOM
    :param image: u_matrix representation of the SOM loaded from pickle
    :param save_labels: bool to save the label matrix
    :param plot: bool to plot diagrams showing the found clusters
    :param wanted_clusters: number that specifies the number of clusters to be found
    :return: matrix of labels each corresponding to a cluster
    """
    n_cluster = []
    scaler = MinMaxScaler(feature_range=(0, 255))
    image = scaler.fit_transform(image)
    lower_tb = 0
    upper_tb = 160
    for threshold in np.linspace(lower_tb, upper_tb, upper_tb - lower_tb+1):
        # thresh = filters.threshold_mean(image)
        binary = image < threshold
        distance = ndi.distance_transform_edt(binary)
        local_maxi = peak_local_max(distance, indices=False, labels=binary)  # , footprint=np.ones((1, 1)))
        markers = ndi.label(local_maxi)[0]
        labels = watershed(distance, markers, mask=binary)
        n_cluster.append(np.unique(labels).size)

    # plt.hist(n_cluster, bins=len(n_cluster))
    # plt.show()
    # liste aus Tupel, erster Eintrag, erster Wert ist die Clusternummer
    if wanted_clusters is None:
        most_common_cluster = Counter(n_cluster).most_common(1)[0][0]
        index_common_cluster = None
        for counter, value in enumerate(n_cluster):
            if value == most_common_cluster:
                index_common_cluster = counter
    else:
        if wanted_clusters not in n_cluster:
            raise ValueError("Anzahl Cluster ungültig, nur Werte von {} bis {}".format(min(n_cluster), max(n_cluster)))

        most_common_cluster = wanted_clusters
        index_common_cluster = None
        for counter, value in enumerate(n_cluster):
            if value == most_common_cluster:
                index_common_cluster = counter
    print(r'Clusters chosen: {}'.format(most_common_cluster))
    ideal_treshold = np.linspace(lower_tb, upper_tb, upper_tb - lower_tb+1)[index_common_cluster]
    filters.try_all_threshold(image)
    plt.show()
    binary = image < filters.threshold_otsu(image) #ideal_treshold
    distance = ndi.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance, indices=False, labels=binary, footprint=np.ones((1, 1)))
    markers = ndi.label(local_maxi)[0]
    final_labels = watershed(distance, markers, mask=binary)
    img, ax = plt.subplots(1, 1)
    ax.imshow(final_labels)
    plt.show(img)
    if save_labels:
        with open('watershed_cluster.pickle', 'wb') as handle:
            pickle.dump(final_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex='all', sharey='all')
        ax = axes.ravel()

        ax[0].imshow(image, cmap='gray', interpolation='nearest')
        ax[0].set_title('U-Matrix')
        ax[1].imshow(-distance, cmap='gray', interpolation='nearest')
        ax[1].set_title('Distances')
        ax[2].imshow(final_labels, cmap=plt.cm.get_cmap('gnuplot', len(np.unique(final_labels))))
        ax[2].set_title('Watershed clusters')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()
    return final_labels


def find_neighbors(i, j):
    """
    Helper function that finds the IDs of the neighbours of the node (i,j)
    :param i: row location on grid/array
    :param j: column location on grid/array
    :return: neighbour locations in a 2D matrix with each row including a x, y location of every node
    """
    # uncomment these lines to include diagonal neighbours
    # neighbors = [[(i, j) for i in range(i-1, i+2)] for j in range(j-1, j+2)]
    # neighbors = np.array(neighbors).reshape(-1, 2)
    neighbors = np.array([[i-1, j], [i+1, j], [i, j-1], [i, j+1]], dtype=np.int32)
    return neighbors


def calc_umatrix(grid, dist_func):
    """
    Function that calculates the U-matrix representation of the SOM model vectors
    :param grid: is the grid provided as a return value from SOM().get_centroids()
    :param dist_func: is the distance metric used for U-matrix calculation
    :return: U-matrix = a 2D Array of the same size as the model vector grid
    """
    grid = np.array(grid)
    result_array = np.zeros(shape=[grid.shape[0], grid.shape[1]])

    locs = [[(i, j) for i in range(grid.shape[0])] for j in range(grid.shape[1])]
    locs = np.array(locs).reshape(-1, 2)

    for loc in locs:
        i, j = loc[0], loc[1]
        neighbors = find_neighbors(i, j)
        # find neighbors outside of actual grid
        min_row = 0
        max_row = grid.shape[0] - 1
        min_col = 0
        max_col = grid.shape[1] - 1
        rows = neighbors[:, 0]
        cols = neighbors[:, 1]
        cond_row = np.logical_and(rows >= min_row, rows <= max_row)
        cond_col = np.logical_and(cols >= min_col, cols <= max_col)

        neighbors = neighbors[np.where(np.logical_and(cond_row == True, cond_col == True))]

        nb_weights = grid[neighbors[:, 0], neighbors[:, 1], :]
        center = grid[i, j, :]
        center = center.reshape(1, -1)
        dist = dist_func(center, nb_weights)
        sum_dist = np.sum(dist)
        result_array[i, j] = sum_dist

    result_array = (result_array - np.min(result_array)) / (np.max(result_array) - np.min(result_array))
    return result_array


def remove_border_label(list_of_labels):
    """
    removes label that identifies the border of the U-matrix from list
    :param list_of_labels: list of labels from which the label of the watershed-lines is removed
    :return: cleaned list
    """
    track_value = list_of_labels[0]
    for i, element in enumerate(list_of_labels):
        if element == 0:
            list_of_labels[i] = track_value
        else:
            track_value = element
    return list_of_labels


def mat_compare_v3(reference, to_adjust, plot_wanted=False):
    """
    Function that adjusts the numbering of a supplied label list to fit a given reference list
    :param reference: List of labels that constitute the reference
    :param to_adjust: List of labels that are to be adjusted to the reference
    :return: adjusted list of the given list to_adjust
    """
    if plot_wanted:
        # Plot both lists before alteration
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(211)
        ax1.set_title('Vorher')
        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
        ax2.set_title('Nachher')
        ax1.plot(reference, 'r-')
        ax1.plot(to_adjust, 'k-', alpha=0.7)

    # define initial values for the loop
    error_array = confusion_matrix(reference, to_adjust)
    blocked_numbers = set()

    # while true, iterate through the maxima of the conf-matr and perform swaps of columns and block used values
    while np.max(error_array) != 0:
        idx = np.where(error_array == np.amax(error_array))
        column = idx[1][0]
        row = idx[0][0]

        if column == row:
            error_array[idx] = 0
            blocked_numbers.add(row)
            continue
        elif row in blocked_numbers or column in blocked_numbers:
            error_array[idx] = 0
            continue
        else:
            blocked_numbers.add(row)
            for counter, value in enumerate(to_adjust):
                if value == row:
                    to_adjust[counter] = column
                if value == column:
                    to_adjust[counter] = row
            error_array = confusion_matrix(reference, to_adjust)
            continue

    if plot_wanted:
        ax2.plot(reference, 'r-')
        ax2.plot(to_adjust, 'k-', alpha=0.5)
        fig.tight_layout()
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')
        plt.show(fig)

    return to_adjust
