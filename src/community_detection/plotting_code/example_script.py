import numpy as np
import VEC_lib
import pickle
import EmbeddingVector_lib
import networkx as nx

# VEC parameters
rw_filename = 'sentences.txt'
emb_filename = 'emb.txt'
num_paths = 10
length_path = 10
emb_dim = 2
k = 5                # Number of negative samples per positive sample
window_size = 8
rand_seed = 1
main_saving_path = ""

# Load your graph here as G
with open("example_graph", 'rb') as pickle_SBM_class:
    SBM_class = pickle.load(pickle_SBM_class)
G = SBM_class.graph

model_w2v = VEC_lib.vec(G, num_paths, length_path, emb_dim, window_size, k, seed=rand_seed, epoch=5, threads=1)
node_list = [str(each_node) for each_node in G.nodes()]
vec_embedding_vectors = np.array(model_w2v[node_list])
if emb_dim > 2:
    print("Dimension:{}".format(vec_embedding_vectors.shape))
    u, s, v = np.linalg.svd(vec_embedding_vectors)
    vec_embedding_vectors = np.dot(u[:, :2], np.diag(s[:2]))
VEC_Embeddings = EmbeddingVector_lib.EmbeddingVectors(vec_embedding_vectors, [0] * 250 + [1] * 250,
                                                      main_saving_path, None, None,
                                                      None, None, None,
                                                      random_state=rand_seed)
VEC_Embeddings.align_embeddings()
VEC_Embeddings.predict_label()
# In case we don't have ground-truth label, we can color-coding according to predict_label instead.
VEC_Embeddings.true_label = VEC_Embeddings.predicted_label
VEC_Embeddings.cal_clustering_goodness()
VEC_Embeddings.cal_variance()
VEC_Embeddings.scatter_plot(0, 2, -1.5, 1.5, contour=False)
VEC_Embeddings.save_class()


# # write down the eigenvalues
# with open("Exp_data/sc_emb_vec_reg/pca_singular_values_sc.txt", 'w') as f_write:
#     f_write.write("largest singular values, community 0:\n")
#     for number, n in enumerate(eval_max_0):
#         f_write.write("n = {} ".format(n_array[number]) + str(n) + '\n')
#     f_write.write("smallest singular values, community 0:\n")
#     for number, n in enumerate(eval_min_0):
#         f_write.write("n = {} ".format(n_array[number]) + str(n) + '\n')
#     f_write.write("largest singular values, community 1:\n")
#     for number, n in enumerate(eval_max_1):
#         f_write.write("n = {} ".format(n_array[number]) + str(n) + '\n')
#     f_write.write("smallest singular values, community 1:\n")
#     for number, n in enumerate(eval_min_1):
#         f_write.write("n = {} ".format(n_array[number]) + str(n) + '\n')
#
# this_title_string = "a = {0}, b = {1}, SC-reg".format(a, b)
#
# legend_list = ['Rand '+str(i+1) for i in range(len(random_seed_array))]
#
# # Plot Eigenvalues: max of the max
# fig, ax = plt.subplots()
# for i in range(len(random_seed_array)):
#     ax.plot(n_array, [max(eval_max_0[x][i], eval_max_1[x][i]) for x in range(len(n_array))], 'D')
# ax.legend(legend_list)
# ax.grid()
# ax.set_title(this_title_string + ", \n first eigenvalue")
# fig.savefig("Figures/embedding_vectors_title/SC/quan_max_max_eval_nuc.png".format(i), format='png')
# plt.close(fig)
#
# # Plot Eigenvalues: max of the min
# fig, ax = plt.subplots()
# for i in range(len(random_seed_array)):
#     ax.plot(n_array, [max(eval_min_0[x][i], eval_min_1[x][i]) for x in range(len(n_array))], 'D')
# ax.legend(legend_list)
# ax.grid()
# ax.set_title(this_title_string + ", \n second eigenvalue")
# fig.savefig("Figures/embedding_vectors_title/SC/quan_max_min_eval_nuc.png".format(i), format='png')
# plt.close(fig)
#
# # Plot Distance between means
# fig, ax = plt.subplots()
# for i in range(len(random_seed_array)):
#     ax.plot(n_array, [mean_dist[x][i] for x in range(len(n_array))], 'D')
# ax.legend(legend_list)
# ax.grid()
# ax.set_title(this_title_string + ", \n distance between means")
# fig.savefig("Figures/embedding_vectors_title/SC/quan_ean_dist_nuc.png".format(i), format='png')
# plt.close(fig)
#
# # Plot range distance ratio
# fig, ax = plt.subplots()
# for i in range(len(random_seed_array)):
#     ax.plot(n_array, [dist_ratio[x][i] for x in range(len(n_array))], 'D')
# ax.legend(legend_list)
# ax.grid()
# ax.set_title(this_title_string + ", \nRatio of Avg range and Distance between means")
# fig.savefig("Figures/embedding_vectors_title/SC/quan_dist_ratio_nuc.png".format(i), format='png')
# plt.close(fig)
#
# # Plot snr
# fig, ax = plt.subplots()
# for i in range(len(random_seed_array)):
#     ax.plot(n_array, [snr[x][i] for x in range(len(n_array))], 'D')
# ax.legend(legend_list)
# ax.grid()
# ax.set_title(this_title_string + ", \n SNR")
# fig.savefig("Figures/embedding_vectors_title/SC/quan_snr_nuc.png".format(i), format='png')
# plt.close(fig)
#
# # write down range distance ratio
# with open("Exp_data/sc_emb_vec_reg/range_distance_ratio_nuc.txt", 'w') as f_write:
#     f_write.write("Projected range distance ratio (SC): \n")
#     for number, list_for_each_n in enumerate(dist_ratio):
#         f_write.write("n = {} ".format(n_array[number]) +
#                       "Avg: {} ".format(np.average(list_for_each_n)) + str(list_for_each_n) + '\n')
#
# # write down snr
# with open("Exp_data/sc_emb_vec_reg/snr_nuc.txt", 'w') as f_write:
#     f_write.write("SNR (SC): \n")
#     for number, list_for_each_n in enumerate(snr):
#         f_write.write("n = {} ".format(n_array[number]) +
#                       "Avg: {:.4f} ".format(np.average(list_for_each_n)) + str(list_for_each_n) + '\n')
#
# # todo: sparse plus low rank (splr) for regularized spectral clustering:
# #  https://projecteuclid.org/download/pdfview_1/euclid.aos/1467894715
# # todo: non-backtracking spectral clustering:
# #  https://www.pnas.org/content/pnas/110/52/20935.full.pdf
