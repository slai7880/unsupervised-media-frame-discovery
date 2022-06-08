from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import scipy


def nmi_score(ground_truth_labels, estimated_labels):
    from sklearn import metrics
    nmi = metrics.normalized_mutual_info_score(ground_truth_labels, estimated_labels, average_method="geometric")
    return nmi


def ccr_score(ground_truth_labels, estimated_labels):
    import scipy.optimize as scipy_opt
    from sklearn import metrics

    n = len(ground_truth_labels)
    assignment_matrix = metrics.confusion_matrix(ground_truth_labels, estimated_labels)
    r, c = scipy_opt.linear_sum_assignment(-1 * assignment_matrix)
    ccr = assignment_matrix[r, c].sum() / n
    # todo: [minor] what is Hungarian algorithm?
    return ccr


def confidence_ellipse(x, y, ax, n_std=2.0, face_color='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    face_color: string
        The color of the ellipse

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional data set.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=face_color,
                      **kwargs)
    print(kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)


class EmbeddingVectors:
    """
    A class that process embedding vectors and generates plots.
    """

    def __init__(self, embedding_vectors,
                 # expected_embeddings,  #todo: Reverse change
                 ground_truth_label,
                 main_directory_path, a, b, regime, method, rand, random_state=None):
        """
        embedding_vectors: numpy array n * dim
        ground_truth_label: n * 1 list
        main_directory_path: string. path from current working directory to the main storing folder
        a: float. Within community edge forming probability
        b: float. Across community edge forming probability
        regime: "linear", "logarithm", "constant"
        method: "SC", "VEC", "ErgoPMI", "NucErgo", "NucVEC"
        rand: int. The index of the current graph realization
        random_state: np.random.random_state
        """
        self.original_embeddings = embedding_vectors
        self.projected_2d_embeddings = embedding_vectors
        # self.expected_embeddings = expected_embeddings  #todo: Reverse change
        # self.projected_2d_expected_embeddings = expected_embeddings  #todo: Reverse change
        if embedding_vectors.shape[-1] > 2:
            print("Reducing dimension from:{}".format(embedding_vectors.shape))
            u, s, v = np.linalg.svd(embedding_vectors)
            self.projected_2d_embeddings = np.dot(u[:, :2], np.diag(s[:2]))
            # u, s, v = np.linalg.svd(expected_embeddings) #todo: Reverse change
            # self.projected_2d_expected_embeddings = np.dot(u[:, :2], np.diag(s[:2]))  #todo: Reverse change
        self.aligned_embeddings = self.projected_2d_embeddings
        # self.aligned_expected_embeddings = self.projected_2d_expected_embeddings #todo: Reverse change
        self.n = embedding_vectors.shape[0]
        self.dim = embedding_vectors.shape[1]
        self.k = len(np.unique(ground_truth_label))
        self.true_label = ground_truth_label
        self.directory = main_directory_path
        self.model_para = {"a": a, "b": b, "Regime": regime, "Method": method, "Rand": rand}
        self.rnd_state = random_state
        self.nodes_by_community = {}
        for group in np.unique(self.true_label):
            self.nodes_by_community[group] = [node for node in range(self.n) if self.true_label[node] == group]
        self.predicted_label = None
        # Comparison metrics:
        # About embedding vectors
        self.mean_dist = None
        self.snr = None
        self.srr = None
        self.wc_cov_matrix = None
        self.wc_cov_eigs = None
        self.ac_cov_matrix = None
        self.ac_cov_eigs = None
        # About clustering
        self.ccr = None
        self.nmi = None

    def align_embeddings(self):
        u, s, v = np.linalg.svd(self.original_embeddings)
        aligned_embeddings = np.dot(self.original_embeddings, v.T)
        mean_0 = np.mean(aligned_embeddings[self.nodes_by_community[0], :], axis=0)
        self.aligned_embeddings = np.sign(mean_0) * aligned_embeddings

    def align_expected_embeddings(self):
        r, _ = scipy.linalg.orthogonal_procrustes(self.aligned_expected_embeddings, self.aligned_embeddings)
        self.aligned_expected_embeddings = np.dot(self.aligned_expected_embeddings, r)

    def predict_label(self):
        """
        Use k-means to cluster the aligned_embeddings to get predicted labels
        """
        self.predicted_label = KMeans(n_clusters=self.k, random_state=self.rnd_state)\
            .fit(self.aligned_embeddings).labels_

    def scatter_plot(self, x_lim_0, x_lim_1, y_lim_0, y_lim_1,
                     mean_of_expected=False, contour=True, legend_loc='upper right'):
        """
        Scatter plots of the embedding vectors
        x_lim: the range of x axis
        y_lim: the range of y axis
        """
        fig, ax = plt.subplots()
        marker_size = 8
        # a_ = self.model_para["a"]
        # b_ = self.model_para["b"]
        # constant = np.sqrt(np.log((3 * a_ + b_) / (3 * b_ + a_)))
        # Temporary modifications: below line
        # constant = 0.22616  # Nuc_expectation boost_ratio = 3
        # constant = 0.18466  # boost_ratio = 2
        # constant = 0.26115  # boost_ratio = 4
        # constant = 0.26386  # boost_ratio = 5
        # constant = 0.26391  # boost_ratio = 6
        for group in np.unique(self.true_label):
            group_x = self.aligned_embeddings[self.nodes_by_community[group], 0]
            group_y = self.aligned_embeddings[self.nodes_by_community[group], 1]
            # ax.scatter(group_x, group_y, s=marker_size, alpha=100 / self.n, linewidths=0)
            ax.scatter(group_x, group_y, s=marker_size, linewidths=0)
            # Plot ellipses and embeddings of expected graphs
            if group == 0:
                # ax.scatter([-1, 1], [1,1],
                #                s=12, marker='x', c='k', linewidths=0, label='Expected graph')
                # if mean_of_expected:
                #     ax.scatter(np.mean(self.aligned_expected_embeddings[self.nodes_by_community[group], 0]),
                #                np.mean(self.aligned_expected_embeddings[self.nodes_by_community[group], 1]),
                #                s=12, marker='x', c='k', linewidths=0, label='Expected graph')
                # else:
                #     ax.scatter(self.aligned_expected_embeddings[self.nodes_by_community[group], 0],
                #                self.aligned_expected_embeddings[self.nodes_by_community[group], 1],
                #                s=12, marker='x', c='k', alpha=100 / self.n, linewidths=0, label='Expected graph')
                if contour:
                    confidence_ellipse(group_x, group_y, ax, n_std=2.45,
                                       edgecolor='red', linewidth=0.75, label="95% contour")
            else:
                # if mean_of_expected:
                #     ax.scatter(np.mean(self.aligned_expected_embeddings[self.nodes_by_community[group], 0]),
                #                np.mean(self.aligned_expected_embeddings[self.nodes_by_community[group], 1]),
                #                s=12, marker='x', c='k', linewidths=0)
                # else:
                #     ax.scatter(self.aligned_expected_embeddings[self.nodes_by_community[group], 0],
                #                self.aligned_expected_embeddings[self.nodes_by_community[group], 1],
                #                s=12, marker='x', c='k', alpha=100 / self.n, linewidths=0)
                if contour:
                    confidence_ellipse(group_x, group_y, ax, n_std=2.45, edgecolor='red', linewidth=0.75)
        ax.grid()
        ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.set_xlim(x_lim_0, x_lim_1)
        ax.set_ylim(y_lim_0, y_lim_1)
        ax.legend(loc=legend_loc)
        if self.model_para["Method"] == "SC" and self.k == 2:
            pass
            # ax.set_xlabel("Second Smallest Eigenvector")
            # ax.set_ylabel("First Smallest Eigenvector")
        else:
            ax.set_xlabel("First SVD component")
            ax.set_ylabel("Second SVD component")
        '''
        # fig.savefig(self.directory + "transparent_scatter_plots/mean_{0}_contour_{1}/png/n{2}_rand{3}.png".
        #             format(mean_of_expected, contour, self.n, self.model_para["Rand"]), format='png')
        fig.savefig(self.directory + "transparent_scatter_plots/png/n{}_rand{}.png".
                                format(self.n, self.model_para["Rand"]), format='png')
        # todo: Reverse change
        # fig.savefig(self.directory + "transparent_scatter_plots/mean_{0}_contour_{1}/pdf/n{2}_rand{3}.pdf".
        #             format(mean_of_expected, contour, self.n, self.model_para["Rand"]), format='pdf')
        fig.savefig(self.directory + "transparent_scatter_plots/pdf/n{}_rand{}.pdf".
                    format(self.n, self.model_para["Rand"]), format='pdf')
        # todo: Reverse change
        plt.close(fig)
        '''
        return fig

    def scatter_plot_new(self, x_lim, y_lim, fig, ax, edge_color, label):
        """
        Scatter plots of the embedding vectors
        x_lim: the range of x axis
        y_lim: the range of y axis
        """
        marker_size = 8
        for group in np.unique(self.true_label):
            group_x = self.aligned_embeddings[self.nodes_by_community[group], 0]
            group_y = self.aligned_embeddings[self.nodes_by_community[group], 1]
            # ax.scatter(group_x, group_y, s=marker_size, alpha=100 / self.n, linewidths=0)
            if group == 0:
                confidence_ellipse(group_x, group_y, ax, n_std=2.45, edgecolor=edge_color, label=label)
            else:
                confidence_ellipse(group_x, group_y, ax, n_std=2.45, edgecolor=edge_color)

    def heat_map_2d(self, x_lim, y_lim, bins, weights):
        """
        2D histogram of the embedding vectors
        bin: number of bins
        weight: the weight of each data point.
        """
        fig, ax = plt.subplots()
        # ax.set_xlim(-x_lim, x_lim)
        # ax.set_ylim(-y_lim, y_lim)
        from astropy.convolution.kernels import Gaussian2DKernel
        from astropy.convolution import convolve
        count_on_mesh_grid, x_bin_edges, y_bin_edges = \
            np.histogram2d(self.aligned_embeddings[:, 0], self.aligned_embeddings[:, 1],
                           bins=[np.linspace(-2, 2, 41), np.linspace(-2, 2, 81)])
        #  ax.imshow(heatmap, cmap="summer")
        ax.imshow(convolve(count_on_mesh_grid, Gaussian2DKernel(x_stddev=2.0)), interpolation='none', cmap="plasma",
                  origin="lower", extent=[-4, 4, -2, 2])
        # ax.set_xticklabels(x_bin_edges)
        # ax.hist2d(self.aligned_embeddings[:, 0], self.aligned_embeddings[:, 1], bins=bins, weights=None, cmap="summer")
        if self.model_para["Method"] == "SC" and self.k == 2:
            ax.set_xlabel("Second Smallest Eigenvector")
            ax.set_ylabel("Third Smallest Eigenvector")
        else:
            ax.set_xlabel("First component")
            ax.set_ylabel("Second component")
        fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap="plasma"), ax=ax)
        ax.grid()
        if self.model_para['Regime'] == 'linear':
            scaling = 1
        elif self.model_para['Regime'] == 'logarithm':
            scaling = np.log(self.n) / self.n
        else:
            scaling = 1 / self.n
        ax.set_title("{0} regime, {1}-reg, a={2}, b={3}, n={4}, \n Intra-prob = {5}, Inter-prob = {6}, Rand {7}".
                     format(self.model_para['Regime'], self.model_para["Method"], self.model_para["a"],
                            self.model_para["b"], self.n, self.model_para["a"] * scaling,
                            self.model_para["b"] * scaling, self.model_para["Rand"]))
        fig.savefig(self.directory + "heat_map_2d/png/n{0}_rand{1}.png".
                    format(self.n, self.model_para["Rand"]), format='png')
        print('png')
        fig.savefig(self.directory + "heat_map_2d/pdf/n{0}_rand{1}.pdf".
                    format(self.n, self.model_para["Rand"]), format='pdf')
        print('pdf')
        plt.close(fig)

    def cal_clustering_goodness(self):
        """
        Only for two clusters. Calculate the signal-noise ratio(SNR) and the signal-range ratio(SRR)
        Definition:
            SNR = \frac{||mu_1 - mu_2||^2}{lambda * sigma_1^2 + (1-lambda) * sigma_2^2}}
            SRR = \frac{||mu_1 - mu_2||^2}{(lambda * range_1 + (1-lambda) * range_2)^2}}
        """
        # Distance between mean of each community
        mean_0 = np.mean(self.aligned_embeddings[self.nodes_by_community[0], :], axis=0)
        mean_1 = np.mean(self.aligned_embeddings[self.nodes_by_community[1], :], axis=0)
        self.mean_dist = np.linalg.norm(mean_0-mean_1)
        # Projection range
        # the unit vector pointing the direction from centroid of cluster 1 to centroid of cluster 2
        centroid_line_direction = (mean_0 - mean_1) / np.linalg.norm(mean_0-mean_1)
        # center the embedding vectors for each cluster
        centered_emb_cluster0 = self.aligned_embeddings[self.nodes_by_community[0], :] - mean_0
        centered_emb_cluster1 = self.aligned_embeddings[self.nodes_by_community[1], :] - mean_1
        emb_projection_cluster0 = np.dot(centered_emb_cluster0, centroid_line_direction)
        emb_projection_cluster1 = np.dot(centered_emb_cluster1, centroid_line_direction)
        range_cluster0 = np.max(emb_projection_cluster0) - np.min(emb_projection_cluster0)
        range_cluster1 = np.max(emb_projection_cluster1) - np.min(emb_projection_cluster1)
        self.srr = self.mean_dist ** 2 / ((range_cluster0 + range_cluster1) * 0.5) ** 2
        self.snr = self.mean_dist ** 2 / (0.5 * np.var(emb_projection_cluster0) + 0.5 * np.var(emb_projection_cluster1))
        # ccr and nmi
        self.ccr = ccr_score(self.true_label, self.predicted_label)
        self.nmi = nmi_score(self.true_label, self.predicted_label)

    def cal_variance(self):
        """
        Calculate the within-cluster and across-cluster covariance matrices and eigenvalues of them.
        """
        #
        self.ac_cov_matrix = np.cov(self.aligned_embeddings.T)
        self.ac_cov_eigs, _ = np.linalg.eigh(self.ac_cov_matrix)
        self.wc_cov_matrix = {}
        for group in np.unique(self.true_label):
            self.wc_cov_matrix[group] = np.cov(self.aligned_embeddings[self.nodes_by_community[group], :].T)
        self.wc_cov_eigs = {}
        for group in np.unique(self.true_label):
            # aa = self.aligned_embeddings[self.nodes_by_community[group], :]
            self.wc_cov_eigs[group], _ = np.linalg.eigh(self.wc_cov_matrix[group])

    def save_class(self):
        """
        Save the entire class. Write embedding vectors into a separate csv file.
        """
        # Save the entire class
        import pickle
        with open(self.directory + "embedding_class_pickle/n{0}_rand{1}_pickle".format(self.n, self.model_para["Rand"]),
                  'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        # Save embedding vectors into csv file
        np.savetxt(self.directory + "csv_files/n{0}_rand{1}.csv".format(self.n, self.model_para["Rand"]),
                   self.aligned_embeddings, delimiter=",", fmt='%.10f')
