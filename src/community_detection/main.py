import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
import VEC_lib
import collections
import EmbeddingVector_lib
from cdlib import algorithms

from common import *

def create_SC_embedding_plots(adjacency_matrix, nodes_by_community, title, fig_filepath):
    SM = SpectralEmbedding(n_components = 2, affinity = 'precomputed', eigen_solver = 'arpack')
    sc_embedding_vectors_2D = SM.fit_transform(adjacency_matrix)
    # aligning
    u, s, v = np.linalg.svd(sc_embedding_vectors_2D)
    sc_embedding_vectors_2D = np.dot(sc_embedding_vectors_2D, v.T)
    means = np.mean(sc_embedding_vectors_2D[nodes_by_community[0], :], axis = 0)
    sc_embedding_vectors_2D = np.sign(means) * sc_embedding_vectors_2D
    
    # return sc_embedding_vectors_2D[:, 0].min(), sc_embedding_vectors_2D[:, 0].max(), sc_embedding_vectors_2D[:, 1].min(), sc_embedding_vectors_2D[:, 1].max()
    
    fig, ax = plt.subplots(figsize = (6, 10))
    marker_size = 8
    for label in nodes_by_community:
        X, Y = sc_embedding_vectors_2D[nodes_by_community[label], 0], sc_embedding_vectors_2D[nodes_by_community[label], 1]
        ax.scatter(X + np.random.uniform(-0.5, 0.5), Y, s = marker_size, alpha = 0.8, linewidth = 0)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if title:
            ax.set_title(title, fontsize = 16)
        ax.set_xlabel("First Spectral Embedding Component", fontsize = 14)
        ax.set_ylabel("Second Spectral Embedding Component", fontsize = 14)
    fig.savefig(fig_filepath + ".png")
    plt.close(fig)
    

def create_VEC_embedding_plots(vec_embedding_vectors, nodes_by_community, title, fig_filepath):
    vec_embedding_vectors_2D = vec_embedding_vectors.copy()
    
    if vec_embedding_vectors_2D.shape[1] > 2:
        u, s, v = np.linalg.svd(vec_embedding_vectors_2D)
        vec_embedding_vectors_2D = np.dot(u[:, :2], np.diag(s[:2]))
    
    # aligning
    u, s, v = np.linalg.svd(vec_embedding_vectors_2D)
    vec_embedding_vectors_2D = np.dot(vec_embedding_vectors_2D, v.T)
    means = np.mean(vec_embedding_vectors_2D[nodes_by_community[0], :], axis = 0)
    vec_embedding_vectors_2D = np.sign(means) * vec_embedding_vectors_2D
    
    fig, ax = plt.subplots(figsize = (6, 10))
    marker_size = 8
    for label in nodes_by_community:
        X, Y = vec_embedding_vectors_2D[nodes_by_community[label], 0], vec_embedding_vectors_2D[nodes_by_community[label], 1]
        ax.scatter(X + np.random.uniform(-0.5, 0.5), Y, s = marker_size, alpha = 0.8, linewidth = 0)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.set_xlim(0, 2)
        ax.set_ylim(-1.5, 1.5)
        if title:
            ax.set_title(title, fontsize = 16)
        ax.set_xlabel("First SVD Component", fontsize = 14)
        ax.set_ylabel("Second SVD Component", fontsize = 14)
    fig.savefig(fig_filepath + ".png")
    plt.close(fig)
    
    # use Christy's code
    '''
    VEC_Embeddings = EmbeddingVector_lib.EmbeddingVectors(vec_embedding_vectors.copy(), [0] * 250 + [1] * 250,
                                                          "", None, None,
                                                          None, None, None,
                                                          random_state = 4)
    VEC_Embeddings.align_embeddings()
    VEC_Embeddings.predict_label()
    # In case we don't have ground-truth label, we can color-coding according to predict_label instead.
    VEC_Embeddings.true_label = VEC_Embeddings.predicted_label
    fig = VEC_Embeddings.scatter_plot(0, 2, -1.5, 1.5, contour=False)
    
    fig.savefig(filepath + "_C.png")
    '''
    

def get_communities(adjacency_matrix, n_community, method, seed = 4, fig_params = None):
    A = adjacency_matrix.copy()
    G = nx.from_numpy_matrix(A)
    is_connected = nx.is_connected(G)
    # print("Connected? {}".format(is_connected))
    if not is_connected:
        n = A.shape[0]
        A = A + 1 / (10 * n) * np.ones((n, n))
        G = nx.from_numpy_matrix(A)
    
    community_labels = None
    if method == "SC":
        SC = SpectralClustering(n_clusters = n_community, affinity = 'precomputed', eigen_solver = 'arpack', random_state = seed)
        community_labels = SC.fit_predict(A)
        if fig_params:
            nodes_by_community = {label : [] for label in range(n_community)}
            for i, label in enumerate(community_labels):
                nodes_by_community[label].append(i)
            create_SC_embedding_plots(A, nodes_by_community, fig_params["title"], fig_params["filepath"])
    elif method == "VEC":
        rw_filename = 'sentences.txt'
        emb_filename = 'emb.txt'
        num_paths = 10
        length_path = 100
        emb_dim = 2
        k = 5                # Number of negative samples per positive sample
        window_size = 4

        model_w2v = VEC_lib.vec(G, num_paths, length_path, emb_dim, window_size, k, seed = seed, epoch = 5, threads = 1)
        node_list = [str(node) for node in G.nodes()]
        vec_embedding_vectors = np.array(model_w2v.wv[node_list])
        
        KM = KMeans(n_clusters = n_community)
        community_labels = KM.fit_predict(vec_embedding_vectors)
        
        if fig_params:
            nodes_by_community = {label : [] for label in range(n_community)}
            for i, label in enumerate(community_labels):
                nodes_by_community[label].append(i)
            create_VEC_embedding_plots(vec_embedding_vectors, nodes_by_community, fig_params["title"], fig_params["filepath"])
    elif method == "WT":
        M = algorithms.walktrap(G).to_node_community_map()
        community_labels = [M[i][0] for i in range(A.shape[0])]
    
    return community_labels

def get_graph_data(graph_dir, conditions):
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    selected_graphs = []
    for graph in graphs:
        select = True
        for key in conditions:
            if conditions[key] != graph[key]:
                select = False
                break
        if select:
            selected_graphs.append(graph)
    return selected_graphs

if __name__ == "__main__":
    news = "GunViolence"
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Similarity Function" : "cosine",
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    output_dir = os.path.join(TABLE_DIR, "community_detection_outputs", news)
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    
    graph_dir = os.path.join(GRAPH_DIR, news)
    graphs = get_graph_data(graph_dir, conditions)
    
    community_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    methods = ["SC", "VEC"]
    parameters = ["FE Existence Base", "FE Existence Coefficient", "FE Existence Similarity Function",
                  "FE Semantics Method", "FE Semantics Coefficient", "FE Selection Scheme"]
    all_parameter_values = {"Parameter" : parameters}
    for i in range(len(graphs)):
        graph = graphs[i]
        
        
        writer = pandas.ExcelWriter(os.path.join(output_dir, "community_labels_" + str(i) + ".xlsx"), engine = "openpyxl")
        values = [graph[p] for p in parameters]
        df_parameters = pandas.DataFrame({"Parameter" : parameters, "Value" : values})
        df_parameters.to_excel(writer, sheet_name = "Parameters", index = False)
        all_parameter_values["Table File " + str(i)] = values
        
        figure_dir = os.path.join(FIGURE_DIR, "community_detection_outputs", news)
        pathlib.Path(figure_dir).mkdir(parents = True, exist_ok = True)
        
        for n_community in community_numbers:
            for method in methods:
                fig_params = {"title" : str(n_community) + "_comms",
                              "filepath" : os.path.join(figure_dir, method + "_" + str(n_community) + "_comms")}
                
                fig_params = None
                
                df_out = pandas.DataFrame({"Framing Element" : graph["Framing Elements"],
                                           "Label" : get_communities(np.array(graph["Adjacency Matrix"]), n_community, method, 4, fig_params)})
                df_out.to_excel(writer, sheet_name = method + "_" + str(n_community) + "_comms", index = False)
        writer.close()
    df_out = pandas.DataFrame(all_parameter_values)
    df_out.to_csv(os.path.join(output_dir, "all_parameter_values.csv"), index = False)