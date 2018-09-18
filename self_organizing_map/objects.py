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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler

from self_organizing_map.utility_funcs import get_watershed, calc_umatrix, remove_border_label


class SOM(object):
    """
    1D or 2D Self-organizing map with sequential learning algorithm with monotonously decreasing learning rate
    and neighbourhood radius
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None,
                 wanted_clusters=15, metric='manhattan', clustering_method='agg',
                 pca_init_wanted=False):
        """
        :param m: number of rows of the map
        :param n: number of columns of the map
        :param dim: dimensionality of the data to process
        :param n_iterations: integer number of epochs that are trained
        :param alpha: learning rate
        :param sigma: initial neighbourhood radius
        :param wanted_clusters: integer number of clusters wanted for clustering
        :param metric: distance metric used to identify the best-matching unit
        :param clustering_method: string determining the algorithm used for clustering
        :param pca_init_wanted: bool for using pca initialisation of weight vectors
        """

        # Assign required variables first
        if metric not in ['cosine', 'euclidean', 'manhattan']:
            raise ValueError('Metric must be either cosine, euclidean or manhattan.')
        else:
            self.metric = metric

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
        self.wanted_clusters = wanted_clusters
        self.pca_init_wanted = pca_init_wanted
        self._n_iterations = abs(int(n_iterations))

        self._weightages = None
        self._locations = None
        self._centroid_grid = None
        self.cluster = None
        self._last_bmus = None
        self.umatrix = None
        self.last_bmu_qe = None
        self.state_dependent_qe_dict = {}
        self.clustered_by_watershed = False

        # INITIALIZE GRAPH
        self._graph = tf.Graph()

        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # weight vectors initialised from random distribution
            self._weightage_vects = tf.Variable(tf.random_normal([m * n, dim], seed=666))

            # location of each neuron as row and column
            self._location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

            # PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

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
            if self.metric == 'manhattan':
                distance = tf.reduce_sum(tf.abs(tf.subtract(self._weightage_vects, self._vect_input)), axis=1)
                bmu_index = tf.argmin(distance)
                # debug = tf.norm(tf.subtract(self._weightage_vects, self._vect_input), ord=0.5, keepdims=True)
                # bmu_index = tf.argmin(debug)

            elif self.metric == 'cosine':
                input_1 = tf.nn.l2_normalize(self._weightage_vects, 0)  # todo VALIDATE
                input_2 = tf.nn.l2_normalize(self._vect_input, 0)
                cosine_similarity = tf.reduce_sum(tf.multiply(input_1, input_2), axis=1)
                # distance = 1.0 - cosine_similarity
                # cosine_distance_op = tf.subtract(1.0, cosine_similarity)
                bmu_index = tf.argmax(cosine_similarity)

            else:
                distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, self._vect_input), 2), 1))
                bmu_index = tf.argmin(distance)

            # This will extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.cast(tf.constant(np.array([1, 2])),
                                                                                     tf.int64)), [2])

            # compute alpha and sigma based on the current iteration
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))

            # this line will compute the decrease of alpha and sigma as a exponential decay:
            # learning_rate_op = tf.exp(tf.negative((6*self._iter_input)/self._n_iterations))

            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning rates for all neurons,
            #  based on iteration number and location in comparison to the BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, bmu_loc), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.multiply(tf.pow(_sigma_op, 2), 2))))
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

    def pca_init(self, input_vects):
        """
        Uses PCA initialization for the weight vectors
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: None
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='randomized')
        cols = self._n
        n_nodes = self._m * self._n
        n_pca_components = 2
        coordinates = np.zeros(shape=(n_nodes, n_pca_components))
        for i in range(n_nodes):
            coordinates[i, 0] = int(i / cols)
            coordinates[i, 1] = int(i % cols)

        maximum = np.max(coordinates, axis=0)

        coordinates = 2 * (coordinates / maximum - 0.5)
        input_mean = np.mean(input_vects, axis=0)
        input_std = np.std(input_vects, axis=0)
        input_vects = (input_vects - input_mean) / input_std

        pca.fit(input_vects)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.linalg.norm(eigvec, axis=1)
        eigvec = ((eigvec.T / norms) * eigval).T
        weight = input_mean + coordinates.dot(eigvec) * input_std

        with self._graph.as_default():
            assign_op = self._weightage_vects.assign(tf.convert_to_tensor(weight, dtype=tf.float32))
        self._sess.run(assign_op)

    def fit(self, input_vects, *args):
        """
        Training of the SOM
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :param args: ignored, added to enable fitting in loop with other sklearn models
        :return: None, adjusts the model weight vectors saved in self._centroid_grid
        """
        if self.pca_init_wanted:
            self.pca_init(input_vects)

        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            t0 = time.time()
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
            t1 = time.time() - t0
            print('EPOCH: {} --- TIME {:.3f}s'.format(iter_no, t1))

        # Store a centroid grid for easy retrieval later on, closing the Session and performing clustering
        centroid_grid = [[] for _ in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._trained = True
        self._sess.close()
        self._get_clusters(n_cluster=self.wanted_clusters)

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

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

    def _get_bmu(self, input_vects, plot_wanted=False):
        """
        Method that determines the location of the BMU for each input vector
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :param plot_wanted: whether the quantization error map should be plotted
        :return: list of x,y locations of the BMU for each input
        """

        start_time = time.time()

        location_vects = [(i, k) for i in range(len(self.get_centroids())) for k in range(len(self.get_centroids()[0]))]
        weightage_vects = np.array(self.get_centroids())
        shape_wv = weightage_vects.shape
        qe_matrix = np.zeros(shape=(shape_wv[0], shape_wv[1]))
        weightage_vects = weightage_vects.reshape((shape_wv[0] * shape_wv[1], shape_wv[-1]))
        to_return = []
        quantization_error = []

        # Distance of the BMU is calculated using the distance function that was given to __init__
        for vect in input_vects:
            vect = vect.reshape(-1, vect.shape[0])
            min_index = self.dist_func(weightage_vects, vect).argmin()
            min_qe = self.dist_func(weightage_vects, vect)[min_index]
            locations = location_vects[min_index]
            qe_matrix[locations[0], locations[1]] = min_qe
            to_return.append(location_vects[min_index])
            quantization_error.append(min_qe)

        self.last_bmu_qe = np.array(quantization_error)
        stop_time = time.time() - start_time

        # prints the duration of the calculation and the average QE over all inputs
        print('Dauer = {:.3f} s'.format(stop_time))
        print('QE = {:.3f}'.format(np.mean(np.array(quantization_error))))

        if plot_wanted:
            qe_map, ax = plt.subplots(1, 1)
            ax.matshow(qe_matrix, cmap='plasma')
            qe_map.show()

        return to_return

    def predict(self, input_vects, learn_qe=True):
        """
        Predicts a cluster label for each input in input_vects based on the clusters in self.cluster
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: list of labels, one for each input in input vects
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")

        if self.clustered_by_watershed:
            self._get_clusters(n_cluster=self.wanted_clusters)
            print('Clustering was by watershed -> changed to {}'.format(self.clustering_method))

        self._last_bmus = self._get_bmu(input_vects)
        label_list = []

        for entry in self._last_bmus:
            label_list.append(self.cluster[entry[0], entry[1]])

        if learn_qe:
            self._state_dependent_qe(label_list)

        return label_list

    def predict_w_umatrix(self, input_vects, learn_qe=True):
        """
        Predicts a label for each input based on the clustering as a result of the watershed transformation
        of the u-matrix
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: list of labels, one for each input
        """
        if not self.clustered_by_watershed:
            self.cluster = get_watershed(self.get_umatrix())
            self.clustered_by_watershed = True
            print('Clustering was by {} -> changed to watershed'.format(self.clustering_method))

        self._last_bmus = self._get_bmu(input_vects)
        label_list = []

        for entry in self._last_bmus:
            label_list.append(self.cluster[entry[0], entry[1]])
        label_list = remove_border_label(label_list)

        if learn_qe:
            self._state_dependent_qe(label_list)

        return label_list

    def fit_transform(self, input_vects):
        """
        Trains the SOM if untrained and returns the BMU coordinates for given input vectors
        :param input_vects: data that is to be process in shape(n_observation, n_features)
        :return: array with x,y coordinate for each input vector
        """
        if not self._trained:
            self.fit(input_vects)

        bmu_list = self._get_bmu(input_vects)
        return np.array(bmu_list, dtype=np.uint8)

    def _get_clusters(self, n_cluster, clustering_method=None, numbers_wanted=True):
        """
        Performs clustering based on sklearn's agglomerative or k-means clustering
        Saves plot of the clustering using imshow
        :param n_cluster: number of desired clusters
        :param clustering_method: string determining the clustering algorithm, either 'agg', 'kmeans'
        :return: the cluster array, also saved in self.cluster
        """
        grid = np.array(self.get_centroids())
        grid_reshaped = np.reshape(grid, (-1, grid.shape[-1]))

        if clustering_method is None:
            clustering_method = self.clustering_method

        if clustering_method == 'agg':
            clusterer = cluster.AgglomerativeClustering(n_clusters=n_cluster, affinity=cosine_distances,
                                                        linkage='average')
        elif clustering_method == 'kmeans':
            clusterer = cluster.KMeans(n_clusters=n_cluster, random_state=42, )
        else:
            raise Exception("Use 'kmeans' for K-Means or 'agg' for agglomerative")

        clusterer.fit(grid_reshaped)
        cluster_array = clusterer.labels_.reshape((grid.shape[0], grid.shape[1]))

        clustermap, ax = plt.subplots(1, 1)
        ax.imshow(cluster_array, cmap='plasma')
        ax.set_title('Clusters = {}, Epochs = {}, Metric = {}'.format(n_cluster,
                                                                      self._n_iterations, self.metric.capitalize()))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        fname = r'{}x{} clusters_{}_epochs_{}_metric_{}.png'.format(self._m, self._n, n_cluster, self._n_iterations,
                                                                    self.metric.capitalize())
        if numbers_wanted:
            used_numbers = []
            for i in range(self._m):
                for j in range(self._n):
                    if cluster_array[i, j] not in used_numbers:
                        ax.text(j, i, cluster_array[i, j], va='center', ha='center', color='w')
                        used_numbers.append(cluster_array[i, j])
                    else:
                        pass

        clustermap.tight_layout()
        self.save_figure(clustermap, fname)
        plt.close(clustermap)

        self.clustered_by_watershed = False
        self.cluster = cluster_array
        return cluster_array

    def get_umatrix(self):
        """
        Calculates the U-matrix representation of the model's weight vectors
        Saves plot of u-matrix
        :return: u-matrix representation as numpy array
        """
        assert self._trained, 'Not trained yet'
        self.umatrix = calc_umatrix(self.get_centroids(), dist_func=self.dist_func)

        umatrix_plot, ax = plt.subplots(1, 1)
        ax.imshow(self.umatrix, cmap='Greys', interpolation='none',
                  vmin=np.min(self.umatrix), vmax=np.max(self.umatrix))
        ax.set_title('{}x{}, Epochs = {}, Metric = {}'.format(self._m, self._n, self._n_iterations,
                                                              self.metric.capitalize()))
        fname = r'{}x{} som_{}_epochs_{}_metric_{}.png'.format(self._m, self._n, self.__class__.__name__,
                                                               self._n_iterations, self.metric.capitalize())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        umatrix_plot.tight_layout()
        self.save_figure(umatrix_plot, fname)
        plt.close(umatrix_plot)

        return self.umatrix

    def hit_histo(self):
        """
        Plots a hit histogram based on the BMUs of the last predictions and plots it
        :return: None, plots hit histogram
        """
        assert self._last_bmus is not None
        x = [point[0] for point in self._last_bmus]
        y = [point[1] for point in self._last_bmus]
        size_array = np.zeros(self.cluster.shape)
        for _x, _y in zip(x, y):
            size_array[_x][_y] += 1
        s = size_array[x, y]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cluster_map = ax.imshow(self.cluster, cmap='plasma')
        scatter = ax.scatter(x, y, s=s, edgecolors='k', facecolors='none')
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

    def export_grid(self, destination):
        """
        Exports the weight vectors of a trained SOM as pickle file for importing in a different map
        :param destination: path to location of the output file
        :return: None, saves weight vectors as pickle
        """
        import pickle
        assert isinstance(destination, str)
        with open(destination, 'wb') as handle:
            pickle.dump(self._centroid_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Export successful')

    @classmethod
    def import_grid(cls, src, wanted_clusters):
        """
        Inits a dummy SOM whose model vectors are imported from a supplied pickle file
        WARNING the dummy should not be fit() to new values as the size and other learning parameters are not imported
        :param src: path to the source file
        :param wanted_clusters: number of clusters for the clustering of the imported grid
        :return: a SOM instance with imported weight vectors and new clustering
        """
        print('!WARNING the dummy should not be fit() to new values as the size and'
              ' other learning parameters are not imported!')
        import pickle
        assert isinstance(src, str)
        to_return = SOM(2, 2, 100, wanted_clusters=wanted_clusters)
        to_return._trained = True
        with open(src, 'rb') as handle:
            to_return._centroid_grid = pickle.load(handle)
        to_return._get_clusters(to_return.wanted_clusters)
        return to_return

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
        surface = ax.imshow(self.get_umatrix(), cmap='hot', interpolation='bilinear')
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

    def show_3d_pca(self):
        """
        Function that calculates the first three principal components and plots the data along those dims
        NOT RELEVANT FOR FUNCTION
        :return: None
        """
        assert self._trained
        pca = PCA(n_components=5, random_state=42)
        grid = np.array(self.get_centroids())
        grid_reshaped = np.reshape(grid, (-1, grid.shape[-1]))
        components = pca.fit_transform(grid_reshaped)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], marker='o', alpha=0.5,
                   c=components[:, 3], s=components[:, 4], cmap='plasma')
        plt.show(fig)

    def show_mds(self):
        assert self._trained
        grid = np.array(self.get_centroids())
        grid_reshaped = np.reshape(grid, (-1, grid.shape[-1]))
        precomputed = self.dist_func(grid_reshaped)
        mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
        embedded_points = mds.fit_transform(precomputed)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(embedded_points[:, 0], embedded_points[:, 1], marker='o', alpha=0.5)
        plt.show(fig)

    def outlier_detect_iso_forest(self, input_vects):
        assert self._trained
        grid = np.array(self.get_centroids(), dtype=np.float32)
        grid_reshaped = np.reshape(grid, (-1, grid.shape[-1]))
        iso_for = IsolationForest(random_state=42, contamination=0.0)
        iso_for.fit(grid_reshaped)
        outliers = iso_for.predict(input_vects)
        return len(np.where(outliers == -1))

    def _state_dependent_qe(self, label_list):
        """
        Function that calculates the qunatization error specific to each cluster
        The average QE for each state and each sample (1 pump) is saved in a dictionary
        :param label_list: List of the labels of predicted by the SOM
        :return: None, saves in dictionary self.state_dependent_qe_dict
        """
        label_set = set(label_list)
        label_array = np.array(label_list, dtype=np.float32)  # float because qe_array also float
        label_array = np.reshape(label_array, newshape=(-1, 1))
        assert label_array.shape == self.last_bmu_qe.shape
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

        qe_fig, ax = plt.subplots(n_rows, n_cols)
        axes = ax.ravel()
        for count, key in enumerate(sorted(self.state_dependent_qe_dict.keys())):
            axes[count].plot(self.state_dependent_qe_dict[key], c='k', marker='o', label='State {}'.format(int(key)))
            axes[count].legend(loc='upper right', fontsize='small')
        qe_fig.tight_layout()
        plt.show(qe_fig)

    @staticmethod
    def save_figure(fig_id, filename, foldername='images'):
        working_dir = os.getcwd()
        folder_path = os.path.join(working_dir, foldername)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        path = os.path.join(folder_path, filename)
        fig_id.savefig(path, bbox_inches='tight', format='png')
        print('Saving figure in {}'.format(str(os.path.abspath(folder_path))))


class PLSOM(SOM):
    """
    Parameterless implementation of the Self-organizing map algorithm
    Is independent of learning rate, solely depends on the initial neighbourhood size
    Otherwise identical to SOM
    """

    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, sigma=None,
                 wanted_clusters=15, metric='manhattan', clustering_method='agg'):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        # Assign required variables first
        if metric not in ['cosine', 'euclidean', 'manhattan']:
            raise ValueError('Metric must be either cosine, euclidean or manhattan.')
        self.metric = metric

        if self.metric == 'cosine':
            self.dist_func = cosine_distances
        elif self.metric == 'manhattan':
            self.dist_func = manhattan_distances
        else:
            self.dist_func = euclidean_distances

        self.clustering_method = clustering_method
        self.umatrix = None
        self._m = m
        self._n = n
        self.wanted_clusters = wanted_clusters
        self.cluster = None
        self._last_bmus = None

        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        # INITIALIZE GRAPH
        self._graph = tf.Graph()

        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]

            self._weightage_vects = tf.Variable(tf.random_normal([m * n, dim], seed=666))

            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

            # PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vector
            self._vect_input = tf.placeholder(tf.float32, [dim])
            self.previous_r = tf.placeholder(tf.float32, shape=())
            # Iteration number
            self._iter_input = tf.placeholder(tf.float32)

            # CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            if self.metric == 'manhattan':
                distance = tf.reduce_sum(tf.abs(tf.subtract(self._weightage_vects, self._vect_input)), axis=1)
                min_distance = tf.reduce_min(distance)
                bmu_index = tf.argmin(distance)

            elif self.metric == 'cosine':
                input_1 = tf.nn.l2_normalize(self._weightage_vects, 0)  # todo VALIDATE
                input_2 = tf.nn.l2_normalize(self._vect_input, 0)
                cosine_similarity = tf.reduce_sum(tf.multiply(input_1, input_2), axis=1)
                # distance = 1.0 - cosine_similarity
                # cosine_distance_op = tf.subtract(1.0, cosine_similarity)
                bmu_index = tf.argmax(cosine_similarity)

            else:
                distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, self._vect_input), 2), 1))
                min_distance = tf.reduce_min(distance)
                bmu_index = tf.argmin(distance)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.cast(tf.constant(np.array([1, 2])),
                                                                                     tf.int64)), [2])

            # _____#
            self.r = tf.maximum(min_distance, self.previous_r)
            epsilon_op = tf.div(min_distance, self.r)
            # _____#
            # To compute the alpha and sigma values based on iteration
            # number
            theta_min = 1.0
            # learning_rate_op = tf.exp(tf.negative(tf.div(self._iter_input**2, self._n_iterations)))
            # learning_rate_op = tf.exp(tf.negative(self._iter_input))
            _sigma_op = (sigma - theta_min) * tf.log(1+epsilon_op*(tf.constant(np.e) - 1)) + theta_min
            # tf.multiply(sigma, epsilon_op)

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, bmu_loc), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(epsilon_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
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

    def fit(self, input_vects, *args):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        r_input = 0.0
        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            t0 = time.time()
            for counter, input_vect in enumerate(input_vects):

                    _, r_input = self._sess.run([self._training_op, self.r],
                                                feed_dict={self._vect_input: input_vect,
                                                           self._iter_input: iter_no,
                                                           self.previous_r: r_input})

            t1 = time.time() - t0
            print('EPOCH: {} --- TIME {:.3f}s'.format(iter_no, t1))

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for _ in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._trained = True
        self._sess.close()
        self._get_clusters(n_cluster=self.wanted_clusters)


class RawDataConverter:
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

    def convert_to_stft(self, scaling_wanted=True, norm='manhattan'):
        if norm not in ['manhattan', 'euclidean']:
            raise ValueError('Norm to use must be manhattan or euclidean')

        self.dataframe.reset_index(inplace=True, drop=True)

        if self.axis == 'all':
            if norm == 'manhattan':
                self.dataframe['all'] = (self.dataframe['x'] + self.dataframe['y'] + self.dataframe['z'])
            elif norm == 'euclidean':
                self.dataframe['all'] = (self.dataframe['x']**2 + self.dataframe['y']**2 + self.dataframe['z']**2)**0.5

            self.dataframe['all'] = (self.dataframe['all'] - np.mean(self.dataframe['all'], axis=0))\
                                     / np.std(self.dataframe['all'], axis=0)

        self.dataframe['time'] = 1/self.sampling_freq * self.dataframe.index
        self.dataframe.fillna(value=0, inplace=True)
        self.dataframe['label'] = self.dataframe['label'].astype(np.int32)

        try:
            spec = plt.specgram(self.dataframe[self.axis], NFFT=self.nfft, Fs=self.sampling_freq,
                                noverlap=self.n_overlap)
            plt.close()

            self.time_frames = spec[2]
            time_frames = spec[2]
            spectrogram_data = spec[0].T

            if scaling_wanted:
                spectrogram_data = (spectrogram_data - np.mean(spectrogram_data, axis=0)) / np.std(spectrogram_data,
                                                                                                   axis=0)
            # create label array
            state_list = []
            for counter, frame in enumerate(time_frames):
                if counter == 0:
                    bool_mask = self.dataframe['time'] <= frame
                else:
                    bool_mask = (self.dataframe['time'] <= time_frames[counter]) &\
                                (self.dataframe['time'] > time_frames[counter - 1])
                state = self.dataframe.loc[bool_mask, 'label'].value_counts().idxmax()
                state_list.append(state)
            state_list = np.array(state_list, dtype=np.int32)
            state_list = np.expand_dims(state_list, axis=1)
            array_with_label = np.hstack([spectrogram_data, state_list])

            self.stft_data = array_with_label

            return self.stft_data

        except KeyError:
            print('Dataframe needs columns named "x", "y" and "z" with acceleration values and "label" column.')

    def train_test_gen(self, train_samples, test_samples, path=None):
        if path is None:
            path = self.path

        try:
            if any(i in train_samples for i in test_samples):
                print('WARNING: Train/Test samples have mutual sample numbers')

            train_arrays, test_arrays = [], []

            for i in train_samples:
                sample = 'sample{}.csv'.format(int(i))
                path_for_convert = os.path.join(path, sample)
                self.dataframe = pd.read_csv(path_for_convert, index_col=0)

                train_arrays.append(self.convert_to_stft())

            for i in test_samples:
                sample = 'sample{}.csv'.format(int(i))
                path_for_convert = os.path.join(path, sample)
                self.dataframe = pd.read_csv(path_for_convert, index_col=0)

                test_arrays.append(self.convert_to_stft())

            train_arrays = np.concatenate(train_arrays)
            test_arrays = np.concatenate(test_arrays)
            X_train, y_train = train_arrays[:, :-1], train_arrays[:, -1]
            X_test, y_test = test_arrays[:, :-1], test_arrays[:, -1]
            return X_train, X_test, y_train, y_test

        except IndexError or FileNotFoundError or TypeError:
            print('Check sample numbers (should be iterable) and correctness of path')

    def read_csv(self, path_to_sample, return_norm=True, scale=True, norm='manhattan'):

        file_loc = r'{}'.format(path_to_sample)
        df = pd.read_csv(file_loc)

        return self._convert_df(df=df, return_norm=return_norm, norm=norm, scaling_before_stft=scale)

    def _read_db_in_chunks(self, path_to_db, chunksize):
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

    def _convert_df(self, df, return_norm=True, norm='manhattan', scaling_before_stft=True):
        if norm not in ['manhattan', 'euclidean'] and return_norm:
            raise ValueError('Norm to use must be manhattan or euclidean')

        for axis in ['x', 'y', 'z']:
            assert axis in df.columns

        if df.isnull().values.any():
            initial = len(df)
            df.dropna(inplace=True, axis=0)
            delta = len(df) - initial
            print('DF has NaN values. {} lines of {} dropped.'.format(abs(delta), initial))

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
    def _get_db_names(folder_path):
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
        self.state_changes = {}
        self.state_change_sequence = {}
        self.is_trained = False
        self.out_transitions = {}
        self.state_change_proba = {}
        self.total_durations = {}
        self.state_kde = {}

    @staticmethod
    def get_transitions(array):
        array1 = np.array(array)
        array2 = np.roll(array1, 1)
        transitions = np.cumsum((array1 != array2).astype(int))
        return transitions

    def _get_state_durations(self, data, train=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.uint8)
        transitions = self.get_transitions(data)

        state_dict = {}

        for number in np.unique(transitions):
            state = data[np.where(transitions == number)][0]
            time = data[np.where(transitions == number)].size # * self.time_frame

            try:
                state_dict[state].append(time)
            except KeyError:
                state_dict[state] = [time]

        if train:
            self.n_transitions.append(np.max(transitions) - 1)
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

    def _which_transitions(self, data, train=False):
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
                    self.state_changes[value] += 1
                except KeyError:
                    self.state_changes[value] = 1

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
            '''
            max_value = max(self.state_change_proba.keys())
            num_rows = int((1 + max_value) // 4)
            if num_rows % 4:
                num_rows += 1
            fig, axes = plt.subplots(nrows=num_rows, ncols=4, )
            axes = axes.ravel()
            for ax in axes:
                ax.axis('off')
            for key, value in self.state_change_proba.items():
                axes[int(key)].pie(value.values(), labels=value.keys())
            plt.show()
            '''
            length_array = len(self.state_change_proba.keys())
            probability_matrix = np.zeros(shape=(length_array, length_array), dtype=np.float32)
            for origin, values in self.state_change_proba.items():
                for destination, prob in values.items():
                    probability_matrix[origin, destination] = prob
            probability_matrix[np.where(probability_matrix == np.nan)] = 0

            fig, ax = plt.subplots(1, 1)
            ax.matshow(probability_matrix, cmap='Greys')
            plt.show(fig)
    '''
    def plot_state_change_prob(self, number):
        lst = self.state_change_sequence[number].copy()
        counter = Counter(lst)
        labels_for_pie = []
        sizes_for_pie = []
        for transition, amount in counter.most_common():
            labels_for_pie.append(transition)
            sizes_for_pie.append(amount)

        fig, ax = plt.subplots(1, 1)
        ax.pie(sizes_for_pie, labels=labels_for_pie, autopct='%1.1f%%')
        ax.axis('equal')
        plt.show(fig)
    '''

    def plot_nx_graph(self):
        assert self.is_trained, 'Not trained yet -> no state information'
        self.calc_probability()
        import networkx as nx
        nodes = set([n1 for n1, n2 in self.state_changes.keys()]+[n2 for n1, n2 in self.state_changes.keys()])
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

        loc = nx.circular_layout(G)
        mappable = nx.draw_networkx_nodes(G, loc, nodelist=nodes, node_size=700,
                                          node_color=node_sizes, edgecolors='k', alpha=1.0, label=node_labels,
                                          cmap='coolwarm', vmin=0.0, vmax=1.0)
        nx.draw_networkx_edges(G, loc, width=2.0, arrowsize=14, node_size=500, edge_color='k')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, loc, label_pos=0.2, edge_labels=edge_labels)
        nx.draw_networkx_labels(G, loc)

        plt.colorbar(mappable)
        plt.axis('off')
        plt.show()

    def plot_time_distribution(self):
        assert self.is_trained, 'Not trained yet -> no state information'
        self._generate_kde()

        n_plots = len(self.state_durations.keys())
        n_rows = 4
        n_cols = n_plots//4
        if n_plots % 4:
            n_cols += 1

        duration_fig, ax = plt.subplots(n_rows, n_cols)
        axes = ax.ravel()
        plot_counter = 0

        for key, value_as_array in self.state_durations.items():
            value_as_array = np.array(value_as_array, dtype=np.float32)
            kde = self.state_kde[key]
            start, end = value_as_array.min(), value_as_array.max()
            x_grid = np.linspace(start, end, 50)[:, None]
            curve = np.exp(kde.score_samples(x_grid))
            axes[plot_counter].plot(curve, label='{}'.format(key))
            # sns.distplot(value_as_array, ax=axes[key], rug=False, kde=True, label='State: {}'.format(int(key)))
            axes[plot_counter].set_xlabel('Timeframe')
            axes[plot_counter].legend(loc='upper right', fontsize='small')
            plot_counter += 1

        plt.show(duration_fig)

    def _generate_kde(self):
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

            kde = KernelDensity(bandwidth=1.0)
            kde.fit(value_as_array[:, None])
            # stores the trained KDE for each state
            self.state_kde[key] = kde
        print(self.state_kde)

    def check_total_state_duration(self, data):
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
        print('--- Checking if state changes valid ---')

        state_changes = self._which_transitions(data, train=False)
        if any(change not in self.state_changes.keys() for change in state_changes):
            invalid_changes = [change for change in state_changes if change not in self.state_changes.keys()]
            print('Unknown transitions: {}'.format(len(invalid_changes)))
            for inv in invalid_changes:
                print('Unknown transition {}->{} occurred'.format(inv[0], inv[1]))

    def check_state_remaining_error(self, data):
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


if __name__ == '__main__':
    path_to_files = r'C:\Users\Apex\Desktop\autem_23_07_18\train_data\samples_sensorII_1600hz'
    rdc = RawDataConverter(path=path_to_files, axis='all')
    test_lst = []
    for i in range(4):
        file_n = os.path.join(path_to_files, 'sample{}.csv'.format(i))
        data = rdc.read_csv(file_n)
        test_lst.append(data)
        print('Cycle {}'.format(i))

    X_train = np.concatenate(test_lst[:])

    som1 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=30, alpha=0.3, metric='manhattan')
    som2 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=30, alpha=0.3, metric='euclidean')
    som3 = SOM(m=10, n=5, dim=X_train.shape[1], n_iterations=30, alpha=0.3, metric='cosine')

    results = []
    fig, ax = plt.subplots(3, 1)
    atm = AutomatonV2()
    for count, som in enumerate([som1]):
        som.fit(X_train)
        for db in rdc._get_db_names(r'C:\Users\Apex\Desktop\Kurzversuch_8g'):
            for i in rdc._read_db_in_chunks(path_to_db=db,
                                            chunksize=384000):
                pred = som.predict(i)
                atm.train(pred)
        plt.style.use('ggplot')
        som.plot_state_dependent_qe()
        atm.plot_time_distribution()
        atm.plot_state_durations()
        atm.plot_nx_graph()
        atm.check([0,1,2,3,4,5])
    plt.show(fig)
