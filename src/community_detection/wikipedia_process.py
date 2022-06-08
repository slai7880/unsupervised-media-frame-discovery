#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import VEC_lib
import collections

for method in ["Uniform", "WeightedByArticles", "WeightedByArticlesUnique"]:
    category_file_path = "Wikipedia_category/Data/CategoryRanking{}.csv".format(method)
    df = pd.read_csv(category_file_path)

    for network in ["Articles", "Pages"]:
        network_name = "{}E{}".format(method, network)
        path = "Wikipedia_category/Data/MRanking{}.npy".format(network_name)
        A = np.load(path)
        # Plot Degree distribution
        col_sum = np.sum(A, axis=0)
        col_sum.sort()
        os.makedirs("Wikipedia_category/Outputs/{}".format(network_name), exist_ok=True)
        ax = plt.plot(col_sum[::-1])
        plt.xlabel('Nodes')
        plt.ylabel('Node Degree')
        plt.title('Degree Distribution of {}'.format(network_name))
        plt.text(20, max(col_sum) * 0.95, 'Max = {:.2f}'.format(max(col_sum)))
        plt.text(500, np.mean(col_sum), 'Avg = {:.2f}'.format(np.mean(col_sum)))
        plt.ylim(0, max(col_sum) + 1)
        plt.grid()
        plt.savefig("Wikipedia_category/Outputs/{}/degree_dist.png".format(network_name))
        plt.clf()
        # Plot Heatmap
        plt.imshow(A, cmap='Greys', interpolation='None')
        plt.savefig("Wikipedia_category/Outputs/{}/heat_map.png".format(network_name))
        plt.clf()

        # Community Detection
        # Check Connectivity
        G = nx.from_numpy_matrix(A)
        is_connected = nx.is_connected(G)
        print("Connected? {}".format(is_connected))
        if not is_connected:
            n = 1000
            A = A + 1 / (10 * n) * np.ones((n, n))
            G = nx.from_numpy_matrix(A)

        # Community detection: Spectral clustering
        community_list = [2, 3, 5, 10, 20]
        for k in community_list:
            SC = SpectralClustering(n_clusters=k, affinity='precomputed', eigen_solver='arpack',
                                       random_state=1)
            cluster_labels = SC.fit_predict(A)
            print(collections.Counter(cluster_labels))
            df["SC: {}".format(k)] = cluster_labels

        # Community detection: VEC
        rw_filename = 'sentences.txt'
        emb_filename = 'emb.txt'
        num_paths = 10
        length_path = 100
        emb_dim = 2
        k = 5                # Number of negative samples per positive sample
        window_size = 4

        model_w2v = VEC_lib.vec(G, num_paths, length_path, emb_dim, window_size, k, seed=1, epoch=5, threads=1)
        node_list = [str(each_node) for each_node in G.nodes()]
        vec_embedding_vectors = np.array(model_w2v[node_list])

        for k in community_list:
            KM = KMeans(n_clusters=k)
            cluster_labels = KM.fit_predict(vec_embedding_vectors)
            print(collections.Counter(cluster_labels))
            df["VEC: {}".format(k)] = cluster_labels

        file_name_cos = "Wikipedia_category/Outputs/{}/community_results.xlsx".format(network_name)
        df.to_excel(file_name_cos, header=True, index=False)
