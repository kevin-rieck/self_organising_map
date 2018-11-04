import os
import sqlite3
import time
from collections import Counter

import matplotlib.animation as animation
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler

from self_organizing_map.utility_funcs import get_watershed, calc_umatrix, remove_border_label, ColorCarrier
# from utility_funcs import get_watershed, calc_umatrix, remove_border_label, ColorCarrier


# TODO for visualization see TensorBoard, for SOM see http://www.ai-junkie.com/ann/som/som1.html


class SOM(object):
    """
    1D or 2D Self-organizing map with sequential learning algorithm with monotonously decreasing learning rate
    and neighbourhood radius
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None,
                 wanted_clusters=15, metric='cosine', clustering_method='agg',
                 decay='lin'):
        """
        :param m: number of rows of the map
        :param n: number of columns of the map
        :param dim: number of features
        :param n_iterations: integer number of epochs that are trained
        :param alpha: learning rate
        :param sigma: initial neighbourhood radius
        :param wanted_clusters: integer number of clusters wanted for clustering (only for kmeans and hierarchical clustering)
        :param metric: distance metric used to identify the best-matching unit
        :param clustering_method: string determining the algorithm used for clustering
        """

        # Assign required variables first
        if metric not in ['cosine', 'euclidean', 'manhattan']:
            raise ValueError('Metric must be either cosine, euclidean or manhattan.')
        else:
            self.metric = metric

        # metric function for calculating U-matrix based on the metric for training
        if self.metric == 'cosine':
            self.dist_func = cosine_distances
        elif self.metric == 'manhattan':
            self.dist_func = manhattan_distances
        else:
            self.dist_func = euclidean_distances

        if clustering_method in ['kmeans', 'agg']:
            self.clustering_method = clustering_method
        else:
            raise Exception("Use 'kmeans' for K-Means or 'agg' for agglomerative")

        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)

        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)

        self._m = m
        self._n = n
        self._alpha = alpha
        self.wanted_clusters = wanted_clusters
        self._n_iterations = abs(int(n_iterations))

        self._weightages = None  # list of trained model vectors
        self._locations = None  # list of neuron locations
        self._centroid_grid = None
        self.cluster = None  # int nd array of clusters with shape of som
        self._last_bmus = None
        self.umatrix = None
        self.last_bmu_qe = None  # list of qes of the last input list
        self.state_dependent_qe_dict = {}  # dict of the active states and their mean qe for the last input list
        self.clustered_by_watershed = False  # bool if clustered by watershed
        self.colormap = ColorCarrier().make_cmap('white', 'black')
        # INITIALIZE GRAPH
        self._graph = tf.Graph()

        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # weight vectors initialised from random distribution
            self._weightage_vects = tf.Variable(tf.random_uniform([m * n, dim], minval=0.0, maxval=1.0, seed=666))

            # location of each neuron as row and column
            self._location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

            # PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they will be fed in during training

            # The training vector
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.placeholder("float")

            # CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the distance between every
            # neuron's weight vector and the input, and returns the
            # index of the neuron which gives the least value

            # Based on the chosen metric the distance function will be assigned to self.distance-op
            if self.metric == 'manhattan':
                distance = tf.reduce_sum(tf.abs(tf.subtract(self._weightage_vects, self._vect_input)), axis=1)
                bmu_index = tf.argmin(distance, 0)
                self.distance = tf.reduce_min(distance)

            elif self.metric == 'cosine':
                input_1 = tf.nn.l2_normalize(self._weightage_vects, 1)
                input_2 = tf.nn.l2_normalize(self._vect_input, 0)
                input_2_2d = tf.expand_dims(input_2, 1)
                cosine_similarity = tf.reduce_sum(tf.matmul(input_1, input_2_2d), axis=1)
                distance = 1.0 - cosine_similarity
                bmu_index = tf.argmax(cosine_similarity, 0)
                self.distance = tf.reduce_min(distance)

            else:
                distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, self._vect_input), 2), 1))
                bmu_index = tf.argmin(distance, 0)
                self.distance = tf.reduce_min(distance)

            # This will extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            self.bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.cast(tf.constant(np.array([1, 2])),
                                                                                          tf.int64)), [2])

            # calculating other metrics for input based on BMU index for QE evaluation <- independent of self.metric
            self.manhattan_qe = tf.reduce_sum(tf.abs(tf.subtract(self._weightage_vects[bmu_index], self._vect_input)))

            self.euclidean_qe = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects[bmu_index],
                                                                         self._vect_input), 2)))

            # calculating inputs for cosine distance by normalizing 1D tensors
            cos_input_1 = tf.nn.l2_normalize(self._weightage_vects[bmu_index], 0)
            cos_input_2 = tf.nn.l2_normalize(self._vect_input, 0)
            cosine_similarity = tf.reduce_sum(tf.multiply(cos_input_1, cos_input_2))
            self.cosine_qe = 1.0 - cosine_similarity

            if decay == 'exp':
                # this line will compute the decrease of alpha and sigma as a exponential decay:
                learning_rate_op = tf.exp(tf.negative((6*self._iter_input)/self._n_iterations))
            else:
                # compute alpha and sigma linearly based on the current iteration
                learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))

            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning rates for all neurons,
            #  based on iteration number and location in comparison to the BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects, self.bmu_loc), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"),
                                                           tf.multiply(tf.pow(_sigma_op, 2), 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons based on
            #  a particular input
            learning_rate_multiplier = tf.expand_dims(learning_rate_op, -1)
            weightage_delta = tf.multiply(learning_rate_multiplier,
                                          tf.subtract(self._vect_input, self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects, weightage_delta)
            self._training_op = tf.assign(self._weightage_vects, new_weightages_op)

            # INITIALIZE SESSION
            self._sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

            # INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    @staticmethod
    def neuron_locations(m, n):
        """
        Returns the grid of neuron locations
        :param m: rows of the grid
        :param n: columns of the grid
        :return: array of x, y location on the grid for each neuron
        """

        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def fit(self, input_vects, *args):
        """
        Training of the SOM
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :param args: ignored, added to enable fitting in loop with other sklearn models
        :return: None, adjusts the model weight vectors saved in self._centroid_grid
        """

        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            t0 = time.time()
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})  # number is necessary for learning rate
            t1 = time.time() - t0
            print('EPOCH: {} --- TIME {:.3f}s'.format(iter_no, t1))

        # Store a centroid grid for easy retrieval later on, closing the Session and performing clustering
        centroid_grid = [[] for _ in range(self._m)]  # same data as self._weightages but better accessibility
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._trained = True
        self._calc_clusters(n_cluster=self.wanted_clusters)

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    # TODO ev. remove method use fit_transform instead
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return

    def _get_bmu(self, input_vects):
        """
        Method that determines the location of the BMU for each input vector
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: list of x,y locations of the BMU for each input
        """

        start_time = time.time()

        to_return = []
        quantization_error = []

        # Distance of the BMU is calculated using the distance function that was given to __init__
        # feed_dict requires a number for self._iter_input but it has no effect because train op is not called
        for vect in input_vects:
            loc = self._sess.run(self.bmu_loc, feed_dict={self._vect_input: vect,
                                                          self._iter_input: self._n_iterations})
            qe = self._sess.run(self.distance, feed_dict={self._vect_input: vect,
                                                          self._iter_input: self._n_iterations})
            to_return.append(loc)
            quantization_error.append(qe)

        self.last_bmu_qe = np.array(quantization_error)
        stop_time = time.time() - start_time

        # prints the duration of the calculation and the average QE over all inputs
        print('Dauer = {:.3f} s'.format(stop_time))
        print('QE = {:.3f}'.format(np.mean(np.array(quantization_error))))

        return to_return

    def predict(self, input_vects, save_qes=True):
        """
        Predicts a cluster label for each input in input_vects based on the clusters in self.cluster
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: list of labels, one for each input in input vects
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")

        if self.clustered_by_watershed:
            self._calc_clusters(n_cluster=self.wanted_clusters)
            print('Clustering was by watershed -> changed to {}'.format(self.clustering_method))

        self._last_bmus = self._get_bmu(input_vects)
        label_list = []

        for entry in self._last_bmus:
            label_list.append(self.cluster[entry[0], entry[1]])

        if save_qes:
            self._state_dependent_qe(label_list)

        return label_list

    def predict_w_umatrix(self, input_vects, save_qes=True, numbers_on_plot=True):
        """
        Predicts a label for each input based on the clustering as a result of the watershed transformation
        of the u-matrix, saves figure of clustermap
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: list of labels, one for each input
        """
        if not self.clustered_by_watershed:
            self.cluster = get_watershed(self.get_umatrix())
            self.clustered_by_watershed = True
            clustermap, ax = plt.subplots(1, 1, figsize=(11.69, 11.69/2))
            ax.imshow(self.cluster, cmap=self.colormap)
            n_cluster = np.max(self.cluster)
            ax.set_title('Clusters = {}, Epochs = {}, Metric = {}, lr = {}'.format(n_cluster, self._n_iterations,
                                                                                   self.metric.capitalize(),
                                                                                   self._alpha))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            fname = r'{}x{} clusters_{}_epochs_{}_metric_{}_lr_{}_ws.png'.format(self._m, self._n, n_cluster,
                                                                                 self._n_iterations,
                                                                                 self.metric.capitalize(), self._alpha)
            if numbers_on_plot:
                used_numbers = []
                for i in range(self._m):
                    for j in range(self._n):
                        if self.cluster[(i, j)] not in used_numbers:
                            ax.text(j, i, self.cluster[(i, j)], va='top', ha='left', color='w',
                                    bbox={'facecolor': ColorCarrier().faps_colors['green']})
                            used_numbers.append(self.cluster[(i, j)])
                        else:
                            pass
            clustermap.tight_layout()
            self.save_figure(clustermap, fname)

            print('Clustering was by {} -> changed to watershed'.format(self.clustering_method))

        self._last_bmus = self._get_bmu(input_vects)
        label_list = []

        for entry in self._last_bmus:
            label_list.append(self.cluster[entry[0], entry[1]])
        label_list = remove_border_label(label_list)

        if save_qes:
            self._state_dependent_qe(label_list)

        return label_list

    def evaluate_different_qe(self, input_vects, which_qe='manhattan', return_as_nparray=True):
        """
        Calculates QE that is different from the one used during training for an array of input vectors
        :param input_vects: array of input data
        :param which_qe: string denoting the desired QE type, must be manhattan, euclidean or cosine
        :param return_as_nparray: whether to return as array, else list
        :return: either list or array of the wanted QE for each input
        """
        if which_qe not in ['manhattan', 'cosine', 'euclidean']:
            raise ValueError('QE name must be "manhattan", "cosine" or "euclidean"')

        start_time = time.time()

        quantization_error = []

        for vect in input_vects:
            if which_qe == 'manhattan':
                qe = self._sess.run(self.manhattan_qe, feed_dict={self._vect_input: vect,
                                                                  self._iter_input: self._n_iterations})
            elif which_qe == 'cosine':
                qe = self._sess.run(self.cosine_qe, feed_dict={self._vect_input: vect,
                                                               self._iter_input: self._n_iterations})
            else:
                qe = self._sess.run(self.euclidean_qe, feed_dict={self._vect_input: vect,
                                                                  self._iter_input: self._n_iterations})

            quantization_error.append(qe)
        if return_as_nparray:
            quantization_error = np.array(quantization_error)

        stop_time = time.time() - start_time
        print('Dauer = {:.3f} s'.format(stop_time))
        return quantization_error

    # TODO is used for hit histogram and animation -> does not return the states
    def fit_transform(self, input_vects):
        """
        Trains the SOM if untrained and returns the BMU coordinates for given input vectors
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: array with x,y coordinate for each input vector
        """
        if not self._trained:
            self.fit(input_vects)

        bmu_list = self._get_bmu(input_vects)
        return np.array(bmu_list, dtype=np.uint16)

    def _calc_clusters(self, n_cluster, clustering_method=None, numbers_wanted=True):
        """
        Performs clustering based on sklearn's agglomerative or k-means clustering
        Saves plot of the clustering using imshow
        :param n_cluster: number of desired clusters
        :param clustering_method: string determining the clustering algorithm, either 'agg', 'kmeans'
        :param numbers_wanted: bool if image should also show the cluster numbers
        :return: the cluster array, also saved in self.cluster
        """
        grid = np.array(self.get_centroids())
        grid_reshaped = np.reshape(grid, (-1, grid.shape[-1]))

        if clustering_method is None:
            clustering_method = self.clustering_method

        if clustering_method == 'agg':
            clusterer = cluster.AgglomerativeClustering(n_clusters=n_cluster, affinity=self.metric,
                                                        linkage='average')
        elif clustering_method == 'kmeans':
            clusterer = cluster.KMeans(n_clusters=n_cluster, random_state=42, )
        else:
            raise Exception("Use 'kmeans' for K-Means or 'agg' for agglomerative")

        clusterer.fit(grid_reshaped)
        cluster_array = clusterer.labels_.reshape((grid.shape[0], grid.shape[1]))

        clustermap, ax = plt.subplots(1, 1, figsize=(11.69, 11.69/2))
        ax.imshow(cluster_array, cmap=self.colormap)
        ax.set_title('Clusters = {}, Epochs = {}, Metric = {}, lr = {}'.format(n_cluster, self._n_iterations,
                                                                               self.metric.capitalize(),
                                                                               self._alpha))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        fname = r'{}x{} clusters_{}_epochs_{}_metric_{}_lr_{}.png'.format(self._m, self._n, n_cluster,
                                                                          self._n_iterations,
                                                                          self.metric.capitalize(), self._alpha)
        if numbers_wanted:
            used_numbers = []
            for i in range(self._m):
                for j in range(self._n):
                    if cluster_array[(i, j)] not in used_numbers:
                        ax.text(j, i, cluster_array[(i, j)], va='top', ha='left', color='w',
                                bbox={'facecolor': ColorCarrier().faps_colors['green']})
                        used_numbers.append(cluster_array[(i, j)])
                    else:
                        pass

        clustermap.tight_layout()
        self.save_figure(clustermap, fname)
        plt.close(clustermap)

        self.clustered_by_watershed = False
        self.cluster = cluster_array
        return np.copy(cluster_array)

    def get_umatrix(self):
        """
        Calculates the U-matrix representation of the model's weight vectors
        Saves plot of u-matrix
        :return: u-matrix representation as numpy array
        """
        assert self._trained, 'Not trained yet'
        self.umatrix = calc_umatrix(self.get_centroids(), dist_func=self.dist_func)

        umatrix_plot, ax = plt.subplots(1, 1)
        ax.imshow(self.umatrix, cmap=self.colormap, interpolation='none',
                  vmin=np.min(self.umatrix), vmax=np.max(self.umatrix))
        ax.set_title('{}x{}, Epochs = {}, Metric = {}, lr = {}'.format(self._m, self._n, self._n_iterations,
                                                                       self.metric.capitalize(), self._alpha))
        fname = r'{}x{} som_{}_epochs_{}_metric_{}_lr_{}.png'.format(self._m, self._n, self.__class__.__name__,
                                                                     self._n_iterations, self.metric.capitalize(),
                                                                     self._alpha)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        umatrix_plot.tight_layout()
        self.save_figure(umatrix_plot, fname)
        plt.close(umatrix_plot)

        return self.umatrix

    def hit_histo(self):
        """
        Plots a hit histogram based on the BMUs of the last predictions
        Growing circles represent number of hits and the nackground is the clustermap
        :return: None, plots hit histogram
        """
        assert self._last_bmus is not None
        y = [point[0] for point in self._last_bmus]
        x = [point[1] for point in self._last_bmus]
        size_array = np.zeros(self.cluster.shape)
        for _x, _y in zip(x, y):
            size_array[_y][_x] += 1
        s = size_array[y, x]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cluster_map = ax.imshow(self.cluster, cmap=self.colormap)
        scatter = ax.scatter(x, y, s=s, edgecolors='w', facecolors='k', alpha=0.5)

        fname = r'histo_{}x{} som_{}_epochs_{}_metric_{}_lr_{}.png'.format(self._m, self._n, self.__class__.__name__,
                                                                           self._n_iterations, self.metric.capitalize(),
                                                                           self._alpha)
        self.save_figure(fig, fname)
        plt.show(fig)

    def override_clusters(self):
        """
        Overrides a chosen cluster with a new cluster number both based on user input
        :return: None, alters self.cluster
        """
        plt.figure()
        plt.imshow(self.cluster)
        plt.show()

        # User Input für die Clusteränderung
        old_cluster_number = int(input("Alte Clusternummer eingeben (int):"))
        new_cluster_number = int(input("Neue Clusternummer eingeben (int):"))

        possible_values = np.unique(self.cluster)
        if new_cluster_number not in possible_values or old_cluster_number not in possible_values:
            raise ValueError('Clusternummer nicht zulässig, nur Werte von {} bis {}'.format(self.cluster.min(),
                                                                                            self.cluster.max()))

        mask_array = self.cluster == old_cluster_number
        self.cluster[mask_array] = new_cluster_number

        # Neusortierung der geänderten Cluster aufsteigend nach Clusternummer
        incomplete_order = sorted([i for i in np.unique(self.cluster)])
        for counter, value in enumerate(incomplete_order):
            self.cluster[(self.cluster == value)] = counter

        plt.imshow(self.cluster)
        plt.show()

    def animate_bmu_trajectory(self, tail=5):
        """
        Creates animation of the BMU trajectory
        :param tail: number of values that are shown at the same time
        :return: None, creates animation
        """
        assert self._last_bmus is not None
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        surface = ax.imshow(self.get_umatrix(), cmap=self.colormap, interpolation='bilinear')
        hits, = ax.plot([], [], marker='o', ls='dashed', markerfacecolor='r', markeredgecolor='none',
                        color='y', markersize=4.0)
        fig.suptitle('BMU trajectory')
        data = np.array(self._last_bmus)

        def update_trajectory(num, data, hits, tail):
            if num > tail:
                hits.set_data(data[num-tail:num, 1], data[num-tail:num, 0])
            else:
                hits.set_data(data[:num, 1], data[:num, 0])
            return hits,

        trajectory_ani = animation.FuncAnimation(fig, update_trajectory, frames=len(data),
                                                 fargs=(data, hits, tail), interval=100)

        plt.show()

    def _state_dependent_qe(self, label_list):
        """
        Function that calculates the quantization error specific to each cluster
        The average QE for each state and each sample (1 pump) is saved in a dictionary
        :param label_list: List of the labels of predicted by the SOM
        :return: None, saves in dictionary self.state_dependent_qe_dict
        """
        label_set = set(label_list)
        label_array = np.array(label_list, dtype=np.float32)  # float because qe_array also float

        if len(label_array.shape) < 2:
            label_array = np.reshape(label_array, newshape=(-1, 1))
            print('Label array shape changed to {}'.format(label_array.shape))
        if len(self.last_bmu_qe.shape) < 2:
            self.last_bmu_qe = np.reshape(self.last_bmu_qe, newshape=(-1, 1))
            print('Last QE array shape changed to {}'.format(self.last_bmu_qe.shape))

        combined_array = np.hstack((label_array, self.last_bmu_qe))

        for value in label_set:
            debug = combined_array[np.where(combined_array[:, 0] == value)]
            average_qe = np.mean(debug[:, 1])
            try:
                self.state_dependent_qe_dict[value].append(average_qe)
            except KeyError:
                self.state_dependent_qe_dict[value] = [average_qe]

    def plot_state_dependent_qe(self):
        n_plots = len(self.state_dependent_qe_dict.keys())
        n_rows = 4
        n_cols = n_plots//4
        if n_plots % 4:
            n_cols += 1

        qe_fig, ax = plt.subplots(n_rows, n_cols, figsize=(11.69, 11.69))
        axes = ax.ravel()
        for count, key in enumerate(sorted(self.state_dependent_qe_dict.keys())):
            axes[count].plot(self.state_dependent_qe_dict[key], c='k', marker='o', label='State {}'.format(int(key)))
            axes[count].legend(loc='upper right', fontsize='small')
        qe_fig.tight_layout()
        plt.show(qe_fig)

    def plot_cluster_mean_spectrum(self, cluster_number, input_vector=None):
        """
        plot function for comapring input and mean freq spectrum of corresponding bmu
        :param cluster_number: show mean spectrum of this cluster number
        :param input_vector: calc best cluster and calc bmu and plots mean spectrum (ignores cluster_number)
        :return: figure, axes
        """
        plt.close('all')
        centroid_grid = np.array(self.get_centroids())
        grid_shape = centroid_grid.shape
        lines = []
        labels = []
        color_lst = []
        if input_vector is not None:
            loc = self._get_bmu([input_vector])
            bmu_spectrum = centroid_grid[loc[0][0], loc[0][1]]
            lines.append(bmu_spectrum)
            labels.append('Spectrum of BMU')
            color_lst.append(ColorCarrier().faps_colors['green'])
            lines.append(input_vector)
            labels.append('Input spectrum')
            color_lst.append(ColorCarrier().faps_colors['red'])
            cluster_number = self.cluster[loc[0][0], loc[0][1]]

        centroid_grid = np.reshape(centroid_grid, (grid_shape[0]*grid_shape[1], grid_shape[-1]))
        cluster_array = self.cluster.reshape(grid_shape[0]*grid_shape[1], )
        idx = np.where(cluster_array == cluster_number)
        subset = centroid_grid[idx]
        print(subset.shape)

        spectrum = np.mean(subset, axis=0)
        lines.append(spectrum)
        labels.append('Mean spectrum of cluster {}'.format(int(cluster_number)))
        color_lst.append(ColorCarrier().faps_colors['black'])
        x_axis_ticks = np.linspace(0, len(spectrum)-1, 6)

        spectrum_fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        for line, label, col in zip(lines, labels, color_lst):
            axes.plot(line, c=col, label=label)
        axes.set_xticks(ticks=x_axis_ticks)
        axes.set_xticklabels(labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set_xlabel(r'Rel. Frequency $\frac{F}{F_{max}}$', size=16)
        axes.set_ylabel('PSD [dB]', size=16)
        axes.tick_params(axis='both', labelsize=14)
        axes.legend(fontsize=14)
        axes.grid()
        spectrum_fig.tight_layout()
        plt.show(spectrum_fig)
        return spectrum_fig, axes

    @staticmethod
    def save_figure(fig_id, filename, foldername='images'):
        working_dir = os.getcwd()
        folder_path = os.path.join(working_dir, foldername)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        path = os.path.join(folder_path, filename)
        fig_id.savefig(path, bbox_inches='tight', format='png', dpi=600)
        print('Saving figure in {}'.format(str(os.path.abspath(folder_path))))


class RawDataConverter:
    """
    Data processing pipelin in form of an object
    """
    def __init__(self, df=None, path=None, sampling_freq=1600, NFFT=512, n_overlap_factor=0.5, axis='y'):
        if path is None:
            self.dataframe = df
        else:
            self.dataframe = None
            self.path = path
        self.stft_data = None
        self.time_frames = None
        self.nfft = NFFT
        self.n_overlap = int(n_overlap_factor*NFFT)
        self.sampling_freq = sampling_freq
        if axis not in ['x', 'y', 'z', 'all']:
            raise ValueError("Axis should be string: x, y, z or all")
        else:
            self.axis = axis

    def convert_to_stft(self, norm='manhattan'):
        """
        Method for labelled data; does STFT of self.dataframe and returns the transformed data
        :param norm: str 'manhattan', 'euclidean' if STFT should be applied to norm of x,y,z instead of single axis
        :return: stf-transformed data
        """
        if norm not in ['manhattan', 'euclidean']:
            raise ValueError('Norm to use must be manhattan or euclidean')

        self.dataframe.reset_index(inplace=True, drop=True)

        # if all sensor axes are wanted they are transformed using a vector norm
        if self.axis == 'all':
            if norm == 'manhattan':
                self.dataframe['all'] = (self.dataframe['x'] + self.dataframe['y'] + self.dataframe['z'])
            elif norm == 'euclidean':
                self.dataframe['all'] = (self.dataframe['x']**2 + self.dataframe['y']**2 + self.dataframe['z']**2)**0.5

        self.dataframe['time'] = 1/self.sampling_freq * self.dataframe.index
        self.dataframe.fillna(value=0, inplace=True)
        self.dataframe['label'] = self.dataframe['label'].astype(np.int32)

        try:
            if self.axis == 'all':
                spec = plt.specgram(self.dataframe['all'], NFFT=self.nfft, Fs=self.sampling_freq,
                                    noverlap=self.n_overlap)
            else:
                spec = plt.specgram(self.dataframe[self.axis], NFFT=self.nfft, Fs=self.sampling_freq,
                                    noverlap=self.n_overlap)

            plt.show()

            self.time_frames = spec[2]
            time_frames = spec[2]
            spectrogram_data = spec[0].T

            # create label array
            state_list = []
            # loop for each time frame to extract most common label and use it as label for this timeframe
            for counter, frame in enumerate(time_frames):
                if counter == 0:
                    bool_mask = self.dataframe['time'] <= frame
                else:
                    bool_mask = (self.dataframe['time'] <= time_frames[counter]) &\
                                (self.dataframe['time'] > time_frames[counter - 1])
                state = self.dataframe.loc[bool_mask, 'label'].value_counts().idxmax()
                state_list.append(state)

            # concatenating the label array with the STFT data horizontally
            state_list = np.array(state_list, dtype=np.int32)
            state_list = np.expand_dims(state_list, axis=1)
            array_with_label = np.hstack([spectrogram_data, state_list])

            self.stft_data = array_with_label

            return self.stft_data

        except KeyError:
            print('Dataframe needs columns named "x", "y" and "z" with acceleration values and "label" column.')

    def train_test_gen(self, train_samples, test_samples, path=None, scale_before_stft=True):
        """
        X_train, X_test, y_train, y_test = train_test_gen()
        Generates training and test data and the respective labels
        :param scale_before_stft: whether x,y,z values are to be standardized before transformation
        :param train_samples: int number in range(0, maximale samplenummer)
        :param test_samples: int number in range(0, maximale samplenummer)
        :param path: path to folder containing .csv files with labelled accleration data
        :return: X_train, X_test, y_train, y_test
        """
        if path is None:
            path = self.path

        try:
            if any(i in train_samples for i in test_samples):
                print('WARNING: Train/Test samples have mutual sample numbers')

            train_arrays, test_arrays = [], []

            # read in samples as csv with name sample{}.csv and scale if wanted and
            # return concatenated data of all samples
            for i in train_samples:
                sample = 'sample{}.csv'.format(int(i))
                path_for_convert = os.path.join(path, sample)
                self.dataframe = pd.read_csv(path_for_convert, index_col=0)
                self.drop_values(self.dataframe)

                if scale_before_stft:
                    self.dataframe[['x', 'y', 'z']] = StandardScaler().fit_transform(self.dataframe[['x', 'y', 'z']])

                train_arrays.append(self.convert_to_stft())

            for i in test_samples:
                sample = 'sample{}.csv'.format(int(i))
                path_for_convert = os.path.join(path, sample)
                self.dataframe = pd.read_csv(path_for_convert, index_col=0)
                self.drop_values(self.dataframe)

                if scale_before_stft:
                    self.dataframe[['x', 'y', 'z']] = StandardScaler().fit_transform(self.dataframe[['x', 'y', 'z']])

                test_arrays.append(self.convert_to_stft())

            train_arrays = np.concatenate(train_arrays)
            test_arrays = np.concatenate(test_arrays)
            X_train, y_train = train_arrays[:, :-1], train_arrays[:, -1]
            X_test, y_test = test_arrays[:, :-1], test_arrays[:, -1]
            return X_train, X_test, y_train, y_test

        except IndexError or FileNotFoundError or TypeError:
            print('Check sample numbers (should be iterable) and correctness of path')

    def read_csv(self, path_to_sample, return_norm=False, scale=True, norm='manhattan'):
        """
        Method to read in samples without labels and STF-transform them
        :param path_to_sample: complete path to csv file
        :param return_norm: whether norm of x,y,z should be used
        :param scale: scaling of data before STFT
        :param norm: if norm wanted which norm
        :return: STF-transformed data as np.ndarray
        """
        file_loc = r'{}'.format(path_to_sample)
        df = pd.read_csv(file_loc)

        return self._convert_df(df=df, return_norm=return_norm, norm=norm, scaling_before_stft=scale)

    def _read_db_in_chunks(self, path_to_db, chunksize):
        """
        Method for connecting to a DB and reading it in chunks with subsequent STFT of the chunks
        :param path_to_db: path to wanted DB
        :param chunksize: how many rows to read in each chunk
        :return: STF-transformed data in form of a generator
        """
        assert isinstance(path_to_db, str), 'Path must be str'
        if path_to_db[-3:] != '.db':
            path_to_db = path_to_db + '.db'
        db = sqlite3.connect(path_to_db)
        query = 'SELECT * FROM ACC1'
        df_gen = pd.read_sql_query(query, db, chunksize=chunksize)

        print('New generator with chunksize {}'.format(chunksize))
        print('Path: {}'.format(path_to_db))

        for df in df_gen:
            yield self._convert_df(df)

    def _convert_df(self, df, return_norm=False, norm='manhattan', scaling_before_stft=True):
        """
        Method that converts an internal pandas dataframe using STFT
        :param df: pandas DataFrame object to convert, must have x,y,z comluns
        :param return_norm: whether norm of x,y,z should be calculated before STFT of that norm
        :param norm: which norm, 'manhattan', 'euclidean'
        :param scaling_before_stft: whether data should be standard scaled before STFT
        :return: ndarray of STFT data shape(timeframes, features)
        """
        if norm not in ['manhattan', 'euclidean'] and return_norm:
            raise ValueError('Norm to use must be manhattan or euclidean')

        for axis in ['x', 'y', 'z']:
            assert axis in df.columns

        self.drop_values(df)
        '''if df.isnull().values.any():
            initial = len(df)
            df.dropna(inplace=True, axis=0)
            delta = len(df) - initial
            print('DF has NaN values. {} lines of {} dropped.'.format(abs(delta), initial))'''

        if scaling_before_stft:
            df[['x', 'y', 'z']] = StandardScaler().fit_transform(df[['x', 'y', 'z']])

        if return_norm:
            if norm == 'manhattan':
                df['norm'] = (df['x'] + df['y'] + df['z'])

            elif norm == 'euclidean':
                df['norm'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

            spec = plt.specgram(df['norm'], self.nfft, self.sampling_freq, noverlap=self.n_overlap, cmap='Greys')

        else:
            spec = plt.specgram(df[self.axis], self.nfft, self.sampling_freq, noverlap=self.n_overlap, cmap='Greys')

        plt.show()
        to_return = spec[0].T

        return to_return

    @staticmethod
    def drop_values(df):
        if df.isnull().values.any():
            initial = len(df)
            df.dropna(inplace=True, axis=0)
            delta = len(df) - initial
            print('DF has NaN values. {} lines of {} dropped.'.format(abs(delta), initial))

    @staticmethod
    def _get_db_names(folder_path):
        """
        Finding the names of .db files in a directory and its subdirectory (only one level deep!!!)
        :param folder_path: path to directory to search
        :return: list of strings describing paths to .db files in searched directory
        """
        paths = []

        entries = os.scandir(folder_path)

        for entry in entries:
            if entry.is_file() and entry.name[-3:] == '.db':
                db_path = os.path.join(folder_path, entry.name)
                paths.append(db_path)

            elif entry.is_dir():
                new_path = os.path.join(folder_path, entry.name)
                new_entries = os.scandir(new_path)

                for new_entry in new_entries:
                    if new_entry.name[-3:] == '.db':
                        db_path = os.path.join(new_path, new_entry.name)
                        paths.append(db_path)

        return paths


class AutomatonV2:
    def __init__(self):
        # self.time_frame = time_frame
        self.n_transitions = []
        self.state_durations = {}
        self.num_state_changes = {}  # number of occurrences of specific transition
        self.state_change_sequence = {}  # TODO ev remove -> not used
        self.is_trained = False
        self.out_transitions = {}  # key -> start label, value -> destination label
        self.state_change_proba = {}
        self.total_durations = {}
        self.state_kde = {}
        self.colormap = ColorCarrier().make_cmap('white', 'black')

    @staticmethod
    def get_transitions(array):
        """
        Method that returns an array of the transitions of the given array as cumulative sum:
        E.g. [3, 3, 3, 9, 9, 5] -> [1, 1, 1, 2, 2, 3]
        :param array: numpy ndarray with datatype = np.int
        :return: nd array with transitions
        """
        array1 = np.array(array)
        array2 = np.roll(array1, 1)  # shifts the array one position to the right
        transitions = np.cumsum((array1 != array2).astype(int))  # number of transitions at the specific position
        return transitions

    def _get_state_durations(self, data, train=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.uint8)
        transitions = self.get_transitions(data)

        state_dict = {}

        for number in np.unique(transitions):
            state = data[np.where(transitions == number)][0]  # state number for each segment between transitions
            time = data[np.where(transitions == number)].size  # total time frames during which a state is active

            # save number of for each state in dict
            try:
                state_dict[state].append(time)
            except KeyError:
                state_dict[state] = [time]

        if train:
            self.n_transitions.append(np.max(transitions) - 1)  # save total number of transitions of given array

            # save individual state durations and total state durations as number of time frames in dicts:
            for key, value in state_dict.items():
                try:
                    self.state_durations[key].extend(value)
                except KeyError:
                    self.state_durations[key] = value

                try:
                    self.total_durations[key].append(sum(value))
                except KeyError:
                    self.total_durations[key] = [sum(value)]

        else:
            return state_dict

    def plot_state_durations(self):
        """
        Plots total duration of each state that was learned via train() as bar plot with error markers
        :return: None
        """
        assert self.is_trained, 'Not trained yet -> no state information'
        x = self.total_durations.keys()
        y = [np.mean(value) for value in self.total_durations.values()]
        std = [np.std(value) for value in self.total_durations.values()]
        plt.style.use('ggplot')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('State durations')
        ax.set_xlabel('State')
        ax.set_ylabel('Time [s]')
        ax.bar(x, y, facecolor='crimson', edgecolor='k', alpha=0.5, yerr=std)
        plt.show(fig)

    def train(self, data):
        data = np.array(data, dtype=np.uint8)
        self._get_state_durations(data, train=True)
        self._which_transitions(data, train=True)
        self.is_trained = True
        self.calc_probability(plot_wanted=True)
        pass

    def _which_transitions(self, data, train=False):
        """
        Keeps track of which transitions were made in the data
        Eg: Saves how many times a transition tuple(x,y)->number was made
        Saves the sequence of transitions (x, y) -> (y, z) -> (z, x)
        Saves which destinations were reached from which origin: (x, y) and (x, z) dict[x] = [y, z]
        :param data: numpy ndarray with datatype = np.int
        :param train: whether transitions should be learned by class ie. saved in attributes
        :return: list of state change tuples (origin, destination)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.uint8)

        transitions = self.get_transitions(data)
        state_changes = []
        state_lst = []
        for i in np.unique(transitions)[:-1]:
            idx1 = np.where(transitions == i)
            idx2 = np.where(transitions == (i+1))
            old_state = data[idx1][-1]
            new_state = data[idx2][0]
            state_change = (old_state, new_state)
            state_changes.append(state_change)

        if train:
            for counter, value in enumerate(state_changes):
                try:
                    self.num_state_changes[value] += 1
                except KeyError:
                    self.num_state_changes[value] = 1

                try:
                    self.state_change_sequence[counter].append(value)
                except KeyError:
                    self.state_change_sequence[counter] = [value]

                try:
                    self.out_transitions[value[0]].append(value[1])
                except KeyError:
                    self.out_transitions[value[0]] = [value[1]]

        else:
            return state_changes

    def calc_probability(self, plot_wanted=False):
        """
        Calculates transition probability matrix and can plot the results
        :param plot_wanted: whether plot is wanted
        :return: None, saves probability matrix as attribute self.state_change_proba[origin]=dict[6:0.5, 2:0.5]
        """
        assert self.is_trained, 'Not trained yet -> no state information'
        for key, value in self.out_transitions.items():
            tmp_dict = {}
            cntr = Counter(value)
            for next_state, amount in cntr.most_common():
                sum_of_transitions = sum(cntr.values())
                percentage = round(amount/sum_of_transitions, 4)
                tmp_dict[next_state] = percentage
            self.state_change_proba[key] = tmp_dict

        if plot_wanted:

            length_array = max(self.state_change_proba.keys())
            probability_matrix = np.zeros(shape=(length_array, length_array), dtype=np.float32)
            for origin, values in self.state_change_proba.items():
                for destination, prob in values.items():
                    try:
                        probability_matrix[origin, destination] = prob
                    except:
                        pass
            probability_matrix[np.where(probability_matrix == np.nan)] = 0

            fig, ax = plt.subplots(1, 1)
            ax.matshow(probability_matrix, cmap=self.colormap)
            plt.show(fig)

    def plot_nx_graph(self):
        """
        Plots Automaton as network graph
        :return: None, shows plot
        """
        assert self.is_trained, 'Not trained yet -> no state information'
        self.calc_probability()
        import networkx as nx
        custom_cmap = ColorCarrier().make_cmap('green', 'red')
        nodes = set([n1 for n1, n2 in self.num_state_changes.keys()] + [n2 for n1, n2 in self.num_state_changes.keys()])
        node_sizes = []
        edge_sizes = []
        node_labels = [str(name) for name in nodes]
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node)
            node_sizes.append(np.mean(self.total_durations[node]))
        for state, dict in self.state_change_proba.items():
            origin = state
            for dest, proba in dict.items():
                G.add_edge(origin, dest, weight=round(proba*100))
        '''
        for edge, value in self.state_changes.items():
            G.add_edge(edge[0], edge[1])
            edge_sizes.append(value)
        '''
        # scale node sizes
        node_sizes = np.array(node_sizes) / np.sum(node_sizes)

        loc = nx.shell_layout(G)
        mappable = nx.draw_networkx_nodes(G, loc, nodelist=nodes, node_size=1500,
                                          node_color=node_sizes, edgecolors='k', alpha=1.0, label=node_labels,
                                          cmap=custom_cmap, vmin=0.0, vmax=1.0)
        nx.draw_networkx_edges(G, loc, width=2.0, arrowsize=20, node_size=1500, edge_color='k')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx_edge_labels(G, loc, label_pos=0.2, edge_labels=edge_labels)
        nx.draw_networkx_labels(G, loc)

        cbar = plt.colorbar(mappable)
        cbar.set_label('fraction of cycle time', size=16)
        plt.axis('off')
        plt.show()

    def plot_time_distribution(self):
        """
        Plots KDE of the time duration of individual states
        :return: None, plots diagram
        """
        assert self.is_trained, 'Not trained yet -> no state information'
        self._generate_kde()

        n_plots = len(self.state_durations.keys())
        n_cols = 4
        n_rows = n_plots//4
        if n_plots % 4:
            n_rows += 1

        duration_fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        axes = ax.ravel()
        plot_counter = 0

        for key, value_as_array in self.state_durations.items():
            value_as_array = np.array(value_as_array, dtype=np.float32)
            kde = self.state_kde[key]
            start, end = value_as_array.min(), value_as_array.max()
            x_grid = np.linspace(start, end, 50)[:, None]
            curve = np.exp(kde.score_samples(x_grid))  # returns prob density - not log prob density
            axes[plot_counter].plot(curve, label='{}'.format(key), c=ColorCarrier().faps_colors['green'])
            # sns.distplot(value_as_array, ax=axes[plot_counter], rug=False, kde=True,
            # label='State: {}'.format(int(key)))  TODO for additional histogram ev. remove
            axes[plot_counter].set_xlabel('Timeframe')
            axes[plot_counter].set_ylabel('Prob. Density')
            axes[plot_counter].legend(loc='upper right', fontsize='small')
            plot_counter += 1

        plt.show(duration_fig)

        return duration_fig, axes

    def _generate_kde(self, bandwidth=1.0):
        """
        Performs Kernel Density Estimation of the durations of individual states
        Gaussian Kernel and bandwith of 1.0 -> can be adjusted
        :return: None, saves a KDE for each state in self.state_kde dict
        """
        from sklearn.neighbors import KernelDensity

        if self.state_durations is None:
            print('Dictionary with individual state durations is None')
            return

        for key, value in self.state_durations.items():
            # finding optimal bandwidth for each state based on cross validation
            value_as_array = np.array(value)
            # grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 4)}, cv=5)
            # grid.fit(value_as_array[:, None])
            # best_params = grid.best_params_

            # print('State: {}, bandwidth: {}'.format(key, best_params['bandwidth']))

            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(value_as_array[:, None])
            # stores the trained KDE for each state
            self.state_kde[key] = kde
        print(self.state_kde)

    def check_total_state_duration(self, data):
        """
        Checks if the total cumulative state durations are within learned limits
        :param data: data to check: ndarray of integers
        :return: None, prints warnings
        """
        print('--- Checking total state durations ---')

        total_state_duration = self._get_state_durations(data, train=False)
        for key, value in self.total_durations.items():
            mean_duration = np.mean(value)
            std_duration = np.std(value)
            try:
                temp_mean = np.sum(total_state_duration[key])
                diff = temp_mean - mean_duration
                if temp_mean > (mean_duration + 3*std_duration):
                    print('Time for state {} longer than expected: {:.2f}s'.format(key, abs(diff)))
                elif temp_mean < (mean_duration - 3*std_duration):
                    print('Time for state {} shorter than expected: {:.2f}s'.format(key, abs(diff)))
                else:
                    pass
            except KeyError:
                print('Observation has no state {}'.format(key))

    def check_changes_valid(self, data):
        """
        Checks if the transitions made in the supplied sequence are valid according to learned transitions
        :param data: data to check: ndarray of integers
        :return: None, prints warnings
        """
        print('--- Checking if state changes valid ---')

        state_changes = self._which_transitions(data, train=False)
        if any(change not in self.num_state_changes.keys() for change in state_changes):
            invalid_changes = [change for change in state_changes if change not in self.num_state_changes.keys()]
            print('Unknown transitions: {}'.format(len(invalid_changes)))
            for inv in invalid_changes:
                print('Unknown transition {}->{} occurred'.format(inv[0], inv[1]))

    def check_state_remaining_error(self, data):
        """
        Checks if a state exceeds the maximum learned threshold
        :param data: data to check: ndarray of integers
        :return: None, prints warnings
        """
        print('--- Checking checking state remaining error ---')

        state_dict = self._get_state_durations(data=data, train=False)
        for key, value in state_dict.items():
            test_min, test_max = min(value), max(value)
            try:
                if test_max > max(self.state_durations[key]):
                    print('State remaining error: Duration of state {} exceeds learned maximum.'.format(key))
                if test_min < min(self.state_durations[key]):
                    print('Early transition: Duration of state {} smaller than learned minimum'.format(key))

            except KeyError:
                print('No learned maximum times for state {}'.format(key))

    def check(self, data):
        self.check_total_state_duration(data)
        self.check_changes_valid(data)
        self.check_state_remaining_error(data)


# TODO auf fehlende Durchlauefe pruefen ueber utc timestamps - vergleiche ende eines sample mit anfang vom naechsten
# TODO ecentuell neue Methodik bei Ultrasschall Daten konzipieren
# TODO Hinweis: datetime.utcfromtimestamp(ts in samples)
class Splitter:
    """
    Class that splits a .db into samples and saves them as .csv files
    Splitting is based on inactivity for more than 15 seconds detected by a RMS value threshold
    """
    def __init__(self, sampling_rate):
        self.srate = sampling_rate
        self.data_gen = None
        self.marker = None
        self.sample_counter = 0

    def split(self, chunksize, destination_folder, db_path, save=True):
        self._read_in_chunks(chunksize, db_path)
        return self._find_standstill(destination_folder, save=save)

    def _read_in_chunks(self, chunksize, db_path):
        connector = sqlite3.connect(db_path)
        self.data_gen = pd.read_sql_query('SELECT * FROM ACC1', connector, chunksize=int(chunksize))

    def _find_standstill(self, destination_folder, seconds=1, axis='y', threshold=1.0, save=True, DOR=1600,
                         check_for_duration=True):
        STILLSTAND_NUMBER_OF_WINDOWS_THRESHOLD = 18  # idea only one of these stillstand phases per pump
        MIN_DURATION = int(218.75 * DOR)
        MAX_DURATION = int(262.5 * DOR)

        window_len = seconds*self.srate  # length of windows to evaluate

        for chunk in self.data_gen:
            x, y, z = chunk['x'], chunk['y'], chunk['z']
            axis_dict = {'x': x, 'y': y, 'z': z}
            axis_to_use = axis_dict[axis]

            result = []
            counter = 0
            marker = []

            for window in range(len(chunk) // window_len):

                start = window * window_len
                end = (window + 1) * window_len
                indicator = np.std(axis_to_use.values[start:end])
                result.append(np.std(axis_to_use.values[start:end]))

                if indicator < threshold:
                    counter += 1
                else:
                    counter = 0

                if counter == STILLSTAND_NUMBER_OF_WINDOWS_THRESHOLD:
                    print('Standstill detected. Marker appended.')
                    marker.append(start)

            self.marker = marker

            for count, mark in enumerate(marker):
                if count == 0:
                    pass

                else:
                    row_idx_start = marker[count-1]
                    row_idx_end = marker[count]
                    sample = chunk.iloc[row_idx_start:row_idx_end, :]

                    if check_for_duration and (MIN_DURATION < len(sample) < MAX_DURATION):

                        sample_name = r'sample{}.csv'.format(self.sample_counter)
                        path_to_save = os.path.join(destination_folder, sample_name)

                        if not os.path.exists(destination_folder):
                            os.makedirs(destination_folder)

                        if save:
                            sample.to_csv(path_to_save)
                            print('Sample {} saved in {}'.format(self.sample_counter, path_to_save))
                            self.sample_counter += 1
                            # yield len(sample)  # TODO check if necessary
                        else:
                            fig = plt.figure()
                            plt.plot(sample['x'])
                            plt.show()


if __name__ == '__main__':

    # folder = r'D:\MA_data\SensorII_151018-171018_8g\shop_floor_test'
    # destination_folder = r'C:\Users\dokisskalt\PycharmProjects\self_organizing_map'
    #
    # splitter = Splitter()
    # for i in range(97):
    #     db_name = r'shop_floor_test{}.db'.format(i)
    #     db_path = os.path.join(folder, db_name)
    #     splitter.split(3e6, destination_folder, db_path, save=False)

    path_to_files = r'..\data'
    rdc = RawDataConverter(path=path_to_files, axis='y')
    test_lst = []
    for i in range(1):
        file_n = os.path.join(path_to_files, 'sample{}.csv'.format(i))
        data = rdc.read_csv(file_n)
        test_lst.append(data)
        print('Cycle {}'.format(i))

    X_train = np.concatenate(test_lst[:])
    som1 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=30, alpha=0.3, metric='manhattan')
    som1.fit(X_train)
    preds = som1.predict(X_train)
    fig, axes = plt.subplots(3, 1)
    for counter, i in enumerate(['manhattan', 'euclidean', 'cosine']):
        result = som1.evaluate_different_qe(X_train, i)
        axes[counter].plot(result, label=i+' QE')
        axes[counter].legend()
    axes[0].plot(som1.last_bmu_qe, c='crimson', ls='dashed', label='QE from prediction', alpha=0.4)
    axes[0].legend()
    plt.show(fig)
    # som2 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=30, alpha=0.3, metric='euclidean')
    # som3 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=10, alpha=0.3, metric='cosine')
    #
    # results = []
    #
    # atm = AutomatonV2()
    # atm.train([0, 1, 2, 1, 2, 3, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    # for count, som in enumerate([som3]):
    # som.fit(X_train)
    # preds = som.predict_w_umatrix(X_train)
    # atm.train(preds)
    # som.plot_state_dependent_qe()
    # som.plot_cluster_mean_spectrum(4, input_vector=X_train[1000])
    # plt.imshow(som.cluster, cmap=ColorCarrier().make_cmap('yellow', 'red'))
    # plt.show()
    # som.plot_cluster_mean_spectrum(5)
    # atm.plot_time_distribution()
    # atm.plot_state_durations()
    # atm.plot_nx_graph()
    # atm.check([0,1,2,3,4,5])
