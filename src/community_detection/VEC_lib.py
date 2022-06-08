"""This file contains all the functions for VEC node embeddings## From graph to embedding:- vec(g, num_paths, length_path, emb_dim, window_size, negative_sampling_rate, seed=1, epoch=5, threads=4,      in_memory=False, rw_filename='sentences.txt', only_one_walk=False, nodes_weighted=False, weight_list=None)- vec_c_implementation(g, num_paths, length_path, emb_dim, window_size, negative_sampling_rate, seed=1, epoch=5,                       threads=4, rw_filename='sentences.txt', emb_filename='embedding_vectors.txt',                       only_one_walk=False, nodes_weighted=False, weight_list=None)## For heavy computation- vec_heavy_compute_sample_corpus(g, num_paths, length_path, rw_filename='sentences.txt', only_one_walk=False,                                  nodes_weighted=False, weight_list=None)- vec_heavy_compute_embed_from_corpus(rw_filename, emb_dim, win_size, emb_filename)"""import mathimport numpy as npimport numpy.random as nprimport gensim.models.word2vec as w2v############################ Categorical sampling functionsdef alias_setup(probability_array):    """    This function is to help draw random samples from discrete distribution with specific weights, the code were adapted    from the following source:    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/        arguments:    probability_array: the discrete probability    return:    redirect_index, redirect_threshold: lists to assist drawing random samples    """    k = len(probability_array)    redirect_threshold = np.zeros(k)    redirect_index = np.zeros(k, dtype=np.int)    # Sort the data into the outcomes with probabilities that are larger and smaller than 1/K.    smaller = []    larger = []    for initial_index, prob in enumerate(probability_array):        redirect_threshold[initial_index] = k * prob        if redirect_threshold[initial_index] < 1.0:            smaller.append(initial_index)        else:            larger.append(initial_index)    # Loop through and create little binary mixtures that appropriately allocate the larger outcomes over the    # overall uniform mixture.    while len(smaller) > 0 and len(larger) > 0:        small = smaller.pop()        large = larger.pop()        redirect_index[small] = large        redirect_threshold[large] = redirect_threshold[large] - (1.0 - redirect_threshold[small])        if redirect_threshold[large] < 1.0:            smaller.append(large)        else:            larger.append(large)    return redirect_index, redirect_thresholddef alias_draw(redirect_index, redirect_threshold):    """    This function is to help draw random samples from discrete distribution with specific weights, the code were adapted    from the following source:    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/        arguments:    redirect_index, redirect_threshold: generated from alias_setup(probability_array)    return:    a random number ranging from 0 to len(probability_array)    """    k = len(redirect_index)    # Draw an initial index from uniform distribution.    index = int(np.floor(npr.rand()*k))    # Flip a coin to decide, either to keep the initial index, or to be redirected to the associated index.    if npr.rand() < redirect_threshold[index]:        return index    else:        return redirect_index[index]##########################def build_node_alias(g):    """    build dictionary nodes_rw_dict that is easier to generate random walks on g    argument:    g: networkx graph objective    return:    nodes_rw_dict with redirect_index, redirect_threshold for each node created using alias_draw functions    """    nodes = g.nodes()    nodes_rw_dict = {}    for each_node in nodes:        neighbours_array = g[each_node]        entry = {'names': [key for key in neighbours_array]}        weights = [neighbours_array[key]['weight'] for key in neighbours_array]        weight_sum = sum(weights)        entry['weights'] = [i / weight_sum for i in weights]        redirect_index, redirect_threshold = alias_setup(entry['weights'])        entry['redirect_index'] = redirect_index        entry['redirect_threshold'] = redirect_threshold        nodes_rw_dict[each_node] = entry    return nodes_rw_dictdef generate_random_walks(nodes_rw_dict, num_paths, length_path, in_memory=False, filename='sentences.txt',                          only_one_walk=False, nodes_weighted=False, nodes_weight_list=None):    """    arguments:    nodes_rw_dict: random walk dictionary from the build_node_alias    options:    in_memory: Boolean. If true, store the random walks in memory; if false, store them in the given filename    filename: write generated random walks into the given file    only_one_walk: Boolean. If true, start only one random walk starting from a randomly chosen node.                            If false, start num_paths random walks from each node.    nodes_weighted: Boolean. If true, launch random walks proportionally to the given nodes weight;                             If false, launch $num_paths$ random walks per node.    nodes_weight_list: List. The weights of the starting nodes to which random walks will be launched proportionally    """    sentence = []    if not in_memory:        f_write = open(filename, 'w')    nodes = nodes_rw_dict.keys()    # If doing one long random walk    if only_one_walk:        start_node = math.floor(npr.rand() * len(nodes))        walk = [start_node]        for j in range(length_path):            current_node = walk[-1]            next_nodes = nodes_rw_dict[current_node]['names']            if len(next_nodes) < 1:  # no neighbor                break            else:                redirect_index = nodes_rw_dict[current_node]['redirect_index']                redirect_threshold = nodes_rw_dict[current_node]['redirect_threshold']                sampled_category_index = alias_draw(redirect_index, redirect_threshold)                next_node = next_nodes[sampled_category_index]                walk.append(next_node)        walk = [str(x) for x in walk]        if in_memory:            sentence.append(walk)        else:            f_write.write(" ".join(walk) + '\n')    # If doing random walk proportionally to the given nodes weight    elif nodes_weighted:        for start_node in nodes:            for i in range(int(nodes_weight_list[start_node]) * num_paths):                walk = [start_node]                for j in range(length_path):                    current_node = walk[-1]                    next_nodes = nodes_rw_dict[current_node]['names']                    if len(next_nodes) < 1:  # no neighbor                        break                    else:                        redirect_index = nodes_rw_dict[current_node]['redirect_index']                        redirect_threshold = nodes_rw_dict[current_node]['redirect_threshold']                        sampled_category_index = alias_draw(redirect_index, redirect_threshold)                        next_node = next_nodes[sampled_category_index]                        walk.append(next_node)                walk = [str(x) for x in walk]                if in_memory:                    sentence.append(walk)                else:                    f_write.write(" ".join(walk) + '\n')    # If doing ordinary random walks    else:        for start_node in nodes:            for i in range(num_paths):                walk = [start_node]                for j in range(length_path):                    current_node = walk[-1]                    next_nodes = nodes_rw_dict[current_node]['names']                    if len(next_nodes) < 1:  # no neighbor                        break                    else:                        redirect_index = nodes_rw_dict[current_node]['redirect_index']                        redirect_threshold = nodes_rw_dict[current_node]['redirect_threshold']                        sampled_category_index = alias_draw(redirect_index, redirect_threshold)                        next_node = next_nodes[sampled_category_index]                        walk.append(next_node)                walk = [str(x) for x in walk]                if in_memory:                    sentence.append(walk)                else:                    f_write.write(" ".join(walk) + '\n')    if not in_memory:        f_write.close()    return sentence################################def vec_heavy_compute_sample_corpus(g, num_paths, length_path, rw_filename='sentences.txt', only_one_walk=False,                                    nodes_weighted=False, weight_list=None):    print('1 building alias auxiliary functions')    nodes_rw_dict = build_node_alias(g)    print('2 launching random walks')    generate_random_walks(nodes_rw_dict, num_paths, length_path, in_memory=False, filename=rw_filename,                          only_one_walk=only_one_walk, nodes_weighted=nodes_weighted, nodes_weight_list=weight_list)    return# def sbm_embed_from_corpus(rw_filename, emb_dim, window_size):#     print('3 learning word2vec models')#     sentence = w2v.LineSentence(rw_filename)#     model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=window_size, min_count=0, sg=1, negative=5,#                              sample=1e-1, workers=4, iter=3)#     return model_w2vdef vec_heavy_compute_embed_from_corpus(rw_filename, emb_dim, win_size, emb_filename):    import os    print('3 learning word2vec models using C code')    command = './word2vec -train ' + rw_filename + ' -output ' + emb_filename + ' -size ' + str(emb_dim) +\              ' -window ' + str(win_size) + ' -negative 5 -cbow 0 -min-count 0 -iter 5 -sample 1e-1'    os.system(command)    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)    return model_w2vdef vec(g, num_paths, length_path, emb_dim, window_size, negative_sampling_rate, seed=1, epoch=5, threads=4,        in_memory=False, rw_filename='sentences.txt', only_one_walk=False, nodes_weighted=False, weight_list=None):    """    VEC algorithm implemented with gensim package    Different modes:    in_memory: If true, file I/O involved: First write all the random walks to disc, then read to learn embeddings.                It's slower than the "in memory" version, but scales well to very large data sets.               If false, random walks will be saved in memory and directly used to learn embeddings.                It can achieve 3x speed up compare to File I/O approach, but does not scale to very large networks.    only_one_walk: Boolean. If true, start only one random walk starting from a randomly chosen node.                            If false, start num_paths random walks from each node.    nodes_weighted: Boolean. If true, launch random walks proportionally to the given nodes weight;                             If false, launch $num_paths$ random walks per node.    weight_list: List. The weights of the starting nodes to which random walks will be launched proportionally    Inputs:    g: graph    num_paths: number of random walks starting from each node    length_path: length of each random walk    emb_dim: the dimensionality of the aligned_embeddings    window_size: the size of neighborhood between which a co-occurrence of nodes is considered as a positive pair    negative_sampling_rate: the number of negative pairs sampled per positive pair    seed: the random seed of the algorithm. If fully reproducible, $threads=1$ is required.          (Can't control the random seed for multi-thread)    epoch: the number of times the algorithm goes through the data set.    threads: the number of threads computing in parallel    """    # print('1 building alias auxiliary functions')    nodes_rw_dict = build_node_alias(g)    # print('2 launching random walks')    sentence = generate_random_walks(nodes_rw_dict, num_paths, length_path, in_memory=in_memory, filename=rw_filename,                                     only_one_walk=only_one_walk, nodes_weighted=nodes_weighted,                                     nodes_weight_list=weight_list)    if not in_memory:        sentence = w2v.LineSentence(rw_filename)    # print('3 learning word2vec models')    model_w2v = w2v.Word2Vec(sentence, vector_size=emb_dim, window=window_size, min_count=0, sg=1,                             negative=negative_sampling_rate, sample=1e-1, workers=threads, epochs=epoch, seed=seed)#    print '4 saving learned aligned_embeddings'#    model_w2v.save_word2vec_format(emb_filename)    return model_w2v    def vec_c_implementation(g, num_paths, length_path, emb_dim, window_size, negative_sampling_rate, seed=1, epoch=5,                         threads=4, rw_filename='sentences.txt', emb_filename='embedding_vectors.txt',                         only_one_walk=False, nodes_weighted=False, weight_list=None):    """    VEC algorithm implemented with the word2vec C implementation (some computational tricks involved).    This implementation scales well to large data sets. Writing random walks and embeddings to file is mandatory.    Different modes:    only_one_walk: Boolean. If true, start only one random walk starting from a randomly chosen node.                            If false, start num_paths random walks from each node.    nodes_weighted: Boolean. If true, launch random walks proportionally to the given nodes weight;                             If false, launch $num_paths$ random walks per node.    weight_list: List. The weights of the starting nodes to which random walks will be launched proportionally    Inputs:    g: graph    num_paths: number of random walks starting from each node    length_path: length of each random walk    emb_dim: the dimensionality of the aligned_embeddings    window_size: the size of neighborhood between which a co-occurrence of nodes is considered as a positive pair    negative_sampling_rate: the number of negative pairs sampled per positive pair    seed: the random seed of the algorithm. If fully reproducible, $threads=1$ is required.          (Can't control the random seed for multi-thread)    epoch: the number of times the algorithm goes through the data set.    threads: the number of threads computing in parallel    emb_filename: the file to which the learned aligned_embeddings of the nodes are written    """    import os    # print('1 building alias auxiliary functions')    nodes_rw_dict = build_node_alias(g)    # print('2 creating random walks')    generate_random_walks(nodes_rw_dict, num_paths, length_path, in_memory=False, filename=rw_filename,                          only_one_walk=only_one_walk, nodes_weighted=nodes_weighted, nodes_weight_list=weight_list)    # print('3 learning word2vec models using C code')    command = './word2vec -train {0} -output {1} -size {2} -window {3} -negative {4} -cbow 0 -min-count 0 -iter {5} ' \              '-sample 1e-1 -seed {6} -workers {7}'.format(rw_filename, emb_filename, emb_dim, window_size,                                                           negative_sampling_rate, epoch, seed, threads)    os.system(command)    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)    # print('4 saving learned aligned_embeddings')    model_w2v.save_word2vec_format(emb_filename)    return model_w2v