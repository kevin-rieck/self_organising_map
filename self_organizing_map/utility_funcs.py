import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.metrics import accuracy_score
from skimage import filters


class ColorCarrier:
    def __init__(self):
        self.faps_colors = {
            'red': (0.6, 0.0, 0.2),
            'red_light': (1.0, 0.79, 0.86),
            'red_dark': (0.39, 0.0, 0.13),
            'green': (0.59, 0.76, 0.22),
            'green_light': (0.80, 0.87, 0.62),
            'green_dark': (0.36, 0.47, 0.14),
            'blue': (0.16, 0.38, 0.58),
            'blue_light': (0.71, 0.82, 0.92),
            'blue_dark': (0.12, 0.28, 0.42),
            'black': (0.0, 0.0, 0.0),
            'grey1': (0.87, 0.87, 0.87),
            'grey2': (0.7, 0.7, 0.7),
            'grey3': (0.5, 0.5, 0.5),
            'grey4': (0.3, 0.3, 0.3),
            'white': (1.0, 1.0, 1.0),
            'orange': (1.0, 0.6, 0.2),
            'orange_light': (1.0, 0.82, 0.64),
            'orange_dark': (0.78, 0.39, 0.0),
            'yellow': (1.0, 0.8, 0.0),
            'yellow_light': (1.0, 0.92, 0.58),
            'yellow_dark': (0.78, 0.64, 0.0)
        }

    def make_cmap(self, color_a, color_b, n_bins=256):
        cmap = clr.LinearSegmentedColormap.from_list('custom blue', [self.faps_colors[color_a],
                                                                     self.faps_colors[color_b]], N=n_bins)
        return cmap


def get_watershed(image, save_labels=False, plot=False):
    """
    uses watershed transformation to get clusters from the presented U-matrix of the SOM
    :param image: u_matrix representation of the SOM loaded from pickle
    :param save_labels: bool to save the label matrix
    :param plot: bool to plot diagrams showing the found clusters
    :return: matrix of labels each corresponding to a cluster
    """

    filters.try_all_threshold(image)

    binary = image < filters.threshold_otsu(image)
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


def mat_compare_v4(reference, to_adjust):

    reference, to_adjust = np.array(reference), np.array(to_adjust)
    print('Initial accuracy: {:.2f}'.format(accuracy_score(reference, to_adjust)))
    adjusted_array = np.copy(to_adjust)
    change_to = None
    change_from = None
    blocked_numbers = set()

    while True:
        max_number = 0
        break_counter = 0

        for i in np.unique(reference):
            counts = np.bincount(adjusted_array[reference == i])
            most_common, number = np.argmax(counts), np.max(counts)

            if number > max_number and most_common not in blocked_numbers and i != most_common:
                max_number = number
                change_from = most_common
                change_to = i
            else:
                break_counter += 1
                if break_counter == len(np.unique(reference)):
                    print('New accuracy: {:.2f}'.format(accuracy_score(reference, adjusted_array)))
                    return np.copy(adjusted_array)

        adjusted_array[to_adjust == change_from] = change_to
        adjusted_array[to_adjust == change_to] = change_from
        blocked_numbers.add(change_to)

        to_adjust = np.copy(adjusted_array)


if __name__ == '__main__':

    list1 = [0, 1, 1, 2, 2, 2, 3, 3, 3]
    list2 = [3, 9, 9, 9, 2, 1, 2, 2, 2]

    adjusted = mat_compare_v4(list1, list2)
    fig1, axes = plt.subplots(2, 1)
    axes[0].plot(list1, c='k')
    axes[0].plot(list2, c='r')
    axes[1].plot(list1, c='k')
    axes[1].plot(adjusted, c='r')
    plt.show(fig1)
