from common import *
from WikiCategoryGraph import *
import networkx as nx
import community
import partition_networkx
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, mutual_info_score
from BERT_util import *
from gensim.utils import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import gensim
from openpyxl import load_workbook

import itertools

def compute_partition_entropy(labels):
    sizes = {}
    for l in labels:
        sizes[l] = 1 + sizes.get(l, 0)
    sizes = np.array([sizes[l] for l in sizes])
    entropy = -np.dot(sizes / len(labels), np.log(sizes / len(labels))).sum()
    return entropy

def compute_labeling_similarity_helper(G, labels_1, labels_2, family):
    similarity_score = {}
    adjusted_similarity_score = {}
    if family == "pair counting":
        similarity_score["RAND"] = rand_score(labels_1, labels_2)
        adjusted_similarity_score["RAND"] = adjusted_rand_score(labels_1, labels_2)

        methods = ["arithmetic", "geometric", "min", "max"]
        entropy = [compute_partition_entropy(labels_1), compute_partition_entropy(labels_2)]
        for method in methods:
            if method == "arithmetic":
                similarity_score["MI MN"] = mutual_info_score(labels_1, labels_2) / np.mean(entropy)
                adjusted_similarity_score["MI MN"] = adjusted_mutual_info_score(labels_1, labels_2, average_method = method)
            elif method == "geometric":
                similarity_score["MI GMN"] = mutual_info_score(labels_1, labels_2) / sp.stats.mstats.gmean(entropy)
                adjusted_similarity_score["MI GMN"] = adjusted_mutual_info_score(labels_1, labels_2, average_method = method)
            elif method == "min":
                similarity_score["MI Min"] = mutual_info_score(labels_1, labels_2) / min(entropy)
                adjusted_similarity_score["MI Min"] = adjusted_mutual_info_score(labels_1, labels_2, average_method = method)
            else:
                similarity_score["MI Max"] = mutual_info_score(labels_1, labels_2) / max(entropy)
                adjusted_similarity_score["MI Max"] = adjusted_mutual_info_score(labels_1, labels_2, average_method = method)
    elif family == "graph-aware":
        labeling_1 = {i : labels_1[i] for i in range(len(labels_1))}
        labeling_2 = {i : labels_2[i] for i in range(len(labels_2))}
        methods = ["RAND", "Jaccard", "MN", "GMN", "Min", "Max"]
        for method in methods:
            similarity_score[method] = G.gam(labeling_1, labeling_2, method = method.lower(), adjusted = False)
            if method != "Jaccard":
                adjusted_similarity_score[method] = G.gam(labeling_1, labeling_2, method = method.lower())
            else:
                adjusted_similarity_score[method] = None
    return similarity_score, adjusted_similarity_score

def compute_labeling_similarity_between_clustering_algorithms_old(families, approach, news, K, rank_condition, graphs):
    for _, graph in enumerate(graphs):
        base_condition = graph["Network Data"].split("_")[0]
        if base_condition == "Page":
            G = nx.convert_matrix.from_numpy_matrix(np.matrix(graph["Adjacency Matrix"]))
            community_dir = os.path.join(TABLE_DIR, approach, news, "community_detection_results", graph["Network Data"], "Top_" + str(K) + "_nonstop_by_" + rank_condition, graph["FE Concurrence Similarity Function"], "FE_semantics_" + str(graph["FE Semantics Method"]))
            if graph["FE Semantics Weight"]:
                community_dir = os.path.join(community_dir, "semantic_weight_" + str(graph["FE Semantics Weight"]))
            community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels.xlsx"))
            
            output_dir = os.path.join(TABLE_DIR, approach, news, "clustering_similarity", graph["Network Data"], "Top_" + str(K) + "_nonstop_by_" + rank_condition, graph["FE Concurrence Similarity Function"], "FE_semantics_" + str(graph["FE Semantics Method"]))
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            for family in families:
                similarity_scores, adjusted_similarity_scores = {}, {}
                for i in range(len(community_excel_file.sheet_names)):
                    labels_1 = pandas.read_excel(community_excel_file, sheet_name = community_excel_file.sheet_names[i])["Label"].values.tolist()
                    for j in range(i + 1, len(community_excel_file.sheet_names)):
                        labels_2 = pandas.read_excel(community_excel_file, sheet_name = community_excel_file.sheet_names[j])["Label"].values.tolist()
                        similarity_score, adjusted_similarity_score = compute_labeling_similarity_helper(G, labels_1, labels_2, family)
                        for key in similarity_score:
                            if not key in similarity_scores:
                                similarity_scores[key] = np.zeros((len(community_excel_file.sheet_names), len(community_excel_file.sheet_names)))
                            similarity_scores[key][i, j] = similarity_score[key]
                        for key in adjusted_similarity_score:
                            if not key in adjusted_similarity_scores:
                                adjusted_similarity_scores[key] = np.zeros((len(community_excel_file.sheet_names), len(community_excel_file.sheet_names)))
                            adjusted_similarity_scores[key][i, j] = adjusted_similarity_score[key]
                writer = pandas.ExcelWriter(os.path.join(output_dir, family + ".xlsx"), engine = "openpyxl")
                for key in similarity_scores:
                    similarity_scores[key] += similarity_scores[key].T
                    for i in range(len(community_excel_file.sheet_names)):
                        similarity_scores[key][i, i] = 1
                    df = pandas.DataFrame(similarity_scores[key], index = community_excel_file.sheet_names, columns = community_excel_file.sheet_names)
                    df.to_excel(writer, sheet_name = key)
                    
                    if key != "Jaccard":
                        adjusted_similarity_scores[key] += adjusted_similarity_scores[key].T
                        for i in range(len(community_excel_file.sheet_names)):
                            adjusted_similarity_scores[key][i, i] = 1
                    else:
                        adjusted_similarity_scores[key] = [[None] * len(community_excel_file.sheet_names) for _ in range(len(community_excel_file.sheet_names))]
                    df = pandas.DataFrame(adjusted_similarity_scores[key], index = community_excel_file.sheet_names, columns = community_excel_file.sheet_names)
                    df.to_excel(writer, sheet_name = "Adjusted " + key)
                writer.close()

def compute_labeling_similarity_between_clustering_algorithms(graph_dir, community_dir, output_dir):
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    
    # choose the best outcome
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Coefficient" : 1,
                  "FE Existence Similarity Function" : "cosine",
                  "FE Semantics Coefficient" : 1,
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    
    selected_graph = None
    for graph in graphs:
        select = True
        for key in conditions:
            if conditions[key] != graph[key]:
                select = False
                break
        if select:
            selected_graph = graph
            
    # get FE labels
    df_parameters = pandas.read_csv(os.path.join(community_dir, "all_parameter_values.csv"))
    table_index = None
    for i in range(df_parameters.shape[1] - 1):
        parameters = {df_parameters["Parameter"][j] : df_parameters["Table File " + str(i)][j] for j in range(df_parameters.shape[0])}
        select = True
        for key in conditions:
            if str(conditions[key]) != parameters[key]:
                select = False
                break
        if select:
            table_index = i
            break
    
    community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels_" + str(table_index) + ".xlsx"))
    n_communities = [i for i in range(2, 21)]
    rand_indices = []
    for i in n_communities:
        df_SC = pandas.read_excel(community_excel_file, sheet_name = "SC_" + str(i) + "_comms")
        df_VEC = pandas.read_excel(community_excel_file, sheet_name = "VEC_" + str(i) + "_comms")
        labels_1, labels_2 = df_SC["Label"].values.tolist(), df_VEC["Label"].values.tolist()
        rand_indices.append(adjusted_rand_score(labels_1, labels_2))
    
    fig_dir = os.path.join(FIGURE_DIR, "WikiCategoryGraph", "GunViolence", "evaluation")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(n_communities, rand_indices, "o")
    ax.set_xlabel("Number of Communities", fontsize = 16)
    ax.set_ylabel("Adjusted RAND Index Score", fontsize = 16)
    ax.set_xticks(n_communities)
    ax.set_xticklabels(n_communities)
    ax.grid(axis = "y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "AdjustedRANDIndices.png"))
    plt.close(fig)

def compute_labeling_similarity_between_semantics_weights(families, approach, news, K, rank_condition, graphs):
    target_weights = [0.5, 1]
    Gs = []
    community_excel_files = []
    target_similarity_function = "jaccard"
    for graph in graphs:
        base_condition = graph["Network Data"].split("_")[0]
        if base_condition == "Page"\
            and graph["FE Concurrence Similarity Function"] == target_similarity_function\
            and str(graph["FE Semantics Method"]) == "BERT"\
            and graph["FE Semantics Weight"]\
            and graph["FE Semantics Weight"] in target_weights:
            G = nx.convert_matrix.from_numpy_matrix(np.matrix(graph["Adjacency Matrix"]))
            Gs.append(G)
            community_dir = os.path.join(TABLE_DIR, approach, news, "community_detection_results", graph["Network Data"], "Top_" + str(K) + "_nonstop_by_" + rank_condition, graph["FE Concurrence Similarity Function"], "FE_semantics_" + str(graph["FE Semantics Method"]))
            if graph["FE Semantics Weight"]:
                community_dir = os.path.join(community_dir, "semantic_weight_" + str(graph["FE Semantics Weight"]))
            community_excel_files.append(pandas.ExcelFile(os.path.join(community_dir, "community_labels.xlsx")))
    
    output_dir = os.path.join(TABLE_DIR, approach, news, "clustering_similarity", "Page_as_Edge_Base", "Top_" + str(K) + "_nonstop_by_" + rank_condition, target_similarity_function, "FE_semantics_BERT")
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    G = Gs[0] # since graph-aware measurements don't care about the edge weights, either is fine
    
    
    for family in families:
        all_similarity_scores = {}
        for sheet_name in community_excel_files[0].sheet_names:
            df_1 = pandas.read_excel(community_excel_files[0], sheet_name = sheet_name)
            labels_1 = df_1["Label"].values.tolist()
            df_2 = pandas.read_excel(community_excel_files[1], sheet_name = sheet_name)
            labels_2 = df_2["Label"].values.tolist()
            similarity_score, adjusted_similarity_score = compute_labeling_similarity_helper(G, labels_1, labels_2, family)
            for key in similarity_score:
                if not key in all_similarity_scores:
                    all_similarity_scores[key] = []
                all_similarity_scores[key].append([similarity_score[key], adjusted_similarity_score[key]])
        print(output_dir)
        writer = pandas.ExcelWriter(os.path.join(output_dir, family + ".xlsx"), engine = "openpyxl")
        for key in all_similarity_scores:
            df = pandas.DataFrame(all_similarity_scores[key], index = community_excel_files[0].sheet_names, columns = ["Original", "Adjusted"])
            df.to_excel(writer, sheet_name = key)
        writer.close()

def adjust_frames(frames = [f.lower() for f in GV_FRAMES], method = "average"):
    frame_to_adjusted_frame_indices = {}
    adjusted_frames = []
    adjusted_frame_to_frame = {}
    for f in frames:
        frame_to_adjusted_frame_indices[f] = []
        if method == "average":
            splitted = []
            if " " or "/" in f:
                if " " in f:
                    splitted = f.split(" ")
                else:
                    splitted = f.split("/")
            else:
                splitted = [f]
            for i in range(len(splitted)):
                adjusted_frames.append(splitted[i])
                frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
                adjusted_frame_to_frame[splitted[i]] = f
        elif method == "custom":
            if f == "economic consequence":
                adjusted_frames.append("economy")
                adjusted_frame_to_frame[adjusted_frames[-1]] = f
                frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
            elif f == "society/culture":
                adjusted_frames.append("society")
                adjusted_frame_to_frame[adjusted_frames[-1]] = f
                frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
                adjusted_frames.append("culture")
                adjusted_frame_to_frame[adjusted_frames[-1]] = f
                frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
            else:
                adjusted_frames.append(f)
                adjusted_frame_to_frame[adjusted_frames[-1]] = f
                frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
        else:
            adjusted_frames.append(f)
            adjusted_frame_to_frame[adjusted_frames[-1]] = f
            frame_to_adjusted_frame_indices[f].append(len(adjusted_frames) - 1)
    return frame_to_adjusted_frame_indices, adjusted_frame_to_frame
    
    

def get_doc2vec_embeddings(texts, source, model_type, version = "Doc2Vec " + gensim.__version__):
    doc2vec = Doc2Vec.load(os.path.join(GENSIM_DIR, "GensimModels", version, source, model_type, "model.model"))
    tokens = [list(tokenize(text, lower = True, deacc = True)) for text in texts]
    embeddings = [doc2vec.infer_vector(tokens[i]) for i in range(len(tokens))]
    text_to_embedding = {texts[i] : embeddings[i] for i in range(len(texts))}
    return text_to_embedding

def get_word2vec_embeddings(texts):
    path = os.path.join(DATA_DIR, "GoogleWord2Vec", "GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(path, binary = True)
    text_to_embedding = {t : model.get_vector(t) for t in texts}
    return text_to_embedding

def get_embedding_mappings(graph, frames, embedding_method):
    fe_to_embedding, frame_to_embedding = {}, {}
    if embedding_method == "BERT":
        fe_to_embedding = graph["FE Semantics Vectors"]
        frame_embeddings = get_BERT_embeddings(frames, False)
        frame_to_embedding = {frames[i] : frame_embeddings[i] for i in range(len(frames))}
    elif embedding_method == "Doc2Vec Wiki":
        fe_to_embedding = get_doc2vec_embeddings(graph["Framing Elements"], "Wiki-June-2021", "PV-DBOW")
        frame_to_embedding = get_doc2vec_embeddings(frames, "Wiki-June-2021", "PV-DBOW")
    elif embedding_method == "Doc2Vec KaggleNews":
        fe_to_embedding = get_doc2vec_embeddings(graph["Framing Elements"], "KaggleNews", "PV-DBOW")
        frame_to_embedding = get_doc2vec_embeddings(frames, "KaggleNews", "PV-DBOW")
    elif embedding_method == "word2vec":
        fe_to_embedding = get_word2vec_embeddings(graph["Framing Elements"])
        frame_to_embedding = get_word2vec_embeddings(frames)
    return fe_to_embedding, frame_to_embedding
        

def get_cluster_to_frame_mappings(graph_dir, community_dir,
                                    frame_adjusting_method, embedding_method, mapping_method, sheet_name):
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    
    # choose the best outcome
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Coefficient" : 1,
                  "FE Existence Similarity Function" : "cosine",
                  "FE Semantics Coefficient" : 1,
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    
    selected_graph = None
    for graph in graphs:
        select = True
        for key in conditions:
            if conditions[key] != graph[key]:
                select = False
                break
        if select:
            selected_graph = graph
            
    frame_to_adjusted_frame_indices, adjusted_frame_to_frame = adjust_frames(method = frame_adjusting_method)
    
    # FE and frame embeddings
    fe_to_embedding, adjusted_frame_to_embedding = get_embedding_mappings(selected_graph, [*adjusted_frame_to_frame], embedding_method)
    
    # get FE labels
    df_parameters = pandas.read_csv(os.path.join(community_dir, "all_parameter_values.csv"))
    table_index = None
    for i in range(df_parameters.shape[1] - 1):
        parameters = {df_parameters["Parameter"][j] : df_parameters["Table File " + str(i)][j] for j in range(df_parameters.shape[0])}
        select = True
        for key in conditions:
            if str(conditions[key]) != parameters[key]:
                select = False
                break
        if select:
            table_index = i
            break
    community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels_" + str(table_index) + ".xlsx"))
    df_labels = pandas.read_excel(community_excel_file, sheet_name = sheet_name)
    fe_to_labels = {df_labels["Framing Element"][i] : df_labels["Label"][i] for i in range(df_labels.shape[0])}
    
    # get label to FE embedding mappings
    label_to_fes = {}
    label_to_fe_embeddings = {}
    for fe in fe_to_labels:
        label = fe_to_labels[fe]
        if not label in label_to_fe_embeddings:
            label_to_fes[label] = []
            label_to_fe_embeddings[label] = []
        label_to_fes[label].append(fe)
        label_to_fe_embeddings[label].append(fe_to_embedding[fe])
    
    '''
    Joyce_labels = ["Gun Control; Politics",
                    "Economic Consequence",
                    "Society/Culture",
                    "Public Safety",
                    "Society/Culture",
                    "Public Safety",
                    "Public Safety",
                    "Gun Control; Society/Culture",
                    "Society/Culture",
                    "Politics"]
    mappings = {"Label" : sorted([*label_to_fes]), "Joyce's labels" : Joyce_labels}
    '''
    mappings = {"Label" : sorted([*label_to_fes])}
    
    if mapping_method == "centroid":
        for i in range(len(GV_FRAMES)):
            mappings["Frame " + str(i + 1)] = []
            mappings["Frame " + str(i + 1) + " score"] = []
        label_to_centroid = {}
        for label in label_to_fe_embeddings:
            label_to_centroid[label] = np.mean(label_to_fe_embeddings[label], axis = 0)
        
        for label in sorted([*label_to_centroid]):
            centroid = label_to_centroid[label]
            frame_to_similarity = {f.lower() : -sys.maxsize for f in GV_FRAMES}
            for adjusted_frame in adjusted_frame_to_embedding:
                similarity = cosine_similarity((centroid, adjusted_frame_to_embedding[adjusted_frame]))
                frame = adjusted_frame_to_frame[adjusted_frame]
                frame_to_similarity[frame] = max(frame_to_similarity[frame], similarity)
            tuples = [(frame, frame_to_similarity[frame]) for frame in frame_to_similarity]
            tuples.sort(reverse = True, key = lambda x: x[1])
            for i in range(len(tuples)):
                mappings["Frame " + str(i + 1)].append(tuples[i][0])
                mappings["Frame " + str(i + 1) + " score"].append(tuples[i][1])
    elif mapping_method == "voting":
        for i in range(len(GV_FRAMES)):
            mappings["Frame " + str(i + 1)] = []
            mappings["Frame " + str(i + 1) + " score"] = []
        for label in sorted([*label_to_fes]):
            votes = {f.lower() : 0 for f in GV_FRAMES}
            for embedding in label_to_fe_embeddings[label]:
                max_similarity, target_frames = -sys.maxsize, []
                for adjusted_frame in adjusted_frame_to_embedding:
                    similarity = cosine_similarity((embedding, adjusted_frame_to_embedding[adjusted_frame]))
                    if similarity == max_similarity:
                        target_frames.append(adjusted_frame_to_frame[adjusted_frame])
                    elif similarity > max_similarity:
                        max_similarity = similarity
                        target_frames = [adjusted_frame_to_frame[adjusted_frame]]
                target_frames = list(set(target_frames))
                for f in target_frames:
                    votes[f] += 1
            tuples = [(f, votes[f]) for f in votes]
            tuples.sort(reverse = True, key = lambda x: x[1])
            for i in range(len(tuples)):
                mappings["Frame " + str(i + 1)].append(tuples[i][0])
                mappings["Frame " + str(i + 1) + " score"].append(tuples[i][1])
    return mappings
    
    """
    two ways to get embeddings:
    1. BERT
    2. doc2vec
    
    doc2vec seems to be better in this case, as it finally yields some "politics" frames
    but the results don't match Joyce's
    may really need to find a news corpus to train
    
    two ways to handle phrases:
    1. take the mean of the component embeddings
    2. separate them into sub-frames
    
    two ways to match frames:
    1. use centroid
    2. use voting
    
    Need to organize the code for clarity.
    
    So far the comparison has been against Joyce's cluster naming, but we still need to check against BERT's sentence prediction.
    
    No guarantee that 
    """
def map_clusters_to_frames(graph_dir, community_dir, output_dir):
    frame_adjusting_methods = [None, "average", "custom"]
    embedding_methods = ["BERT", "Doc2Vec Wiki", "Doc2Vec KaggleNews"]
    mapping_methods = ["centroid", "voting"]
    method_tuples = list(itertools.product(frame_adjusting_methods, embedding_methods, mapping_methods))
    clustering_methods = ["SC", "VEC"]
    # n_communities = [i for i in range(2, 21)]
    clustering_methods = ["SC"]
    n_communities = [6]
    for cm in clustering_methods:
        for n in n_communities:
            print(cm + "_" + str(n) + "_comms")
            writer = pandas.ExcelWriter(os.path.join(output_dir, "clusters_to_frames_" + cm + "_" + str(n) + "_comms.xlsx"), engine = "openpyxl")
            M = {"Index" : [],
                 "Frame adjusting method" : [],
                 "Embedding method" : [],
                 "Mapping method" : []}
            for i in range(len(method_tuples)):
                M["Index"].append(i)
                frame_adjusting_method, embedding_method, mapping_method = method_tuples[i]
                M["Frame adjusting method"].append(frame_adjusting_method)
                M["Embedding method"].append(embedding_method)
                M["Mapping method"].append(mapping_method)
            df_method = pandas.DataFrame(M)
            df_method.to_excel(writer, sheet_name = "Index", index = False)
            for i in range(len(method_tuples)):
                frame_adjusting_method, embedding_method, mapping_method = method_tuples[i]
                print(method_tuples[i])
                sheet_name = cm + "_" + str(n) + "_comms"
                mapping = get_cluster_to_frame_mappings(graph_dir, community_dir, frame_adjusting_method, embedding_method, mapping_method, sheet_name)
                df_out = pandas.DataFrame(mapping)
                df_out.to_excel(writer, sheet_name = str(i), index = False)
            writer.close()

def compare_Joyce_labels_with_sentence_frames(graph_dir, community_dir, output_dir):
    community_label = [["Gun Control", "Politics"],
                       ["Economic Consequence"],
                       ["Society/Culture"],
                       ["Public Safety"],
                       ["Society/Culture"],
                       ["Public Safety"],
                       ["Public Safety"],
                       ["Gun Control", "Society/Culture"],
                       ["Society/Culture"],
                       ["Politics"]]
    
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    
    # choose the best outcome
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Coefficient" : 1,
                  "FE Existence Similarity Function" : "cosine",
                  "FE Semantics Coefficient" : 1,
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    
    selected_graph = None
    for graph in graphs:
        select = True
        for key in conditions:
            if conditions[key] != graph[key]:
                select = False
                break
        if select:
            selected_graph = graph
    
    existence_vectors = np.array([selected_graph["FE Existence Vectors"][fe] for fe in selected_graph["FE Existence Vectors"]])
    
    sentence_frame_dir = os.path.join(TABLE_DIR, "NYT_Tagger", "GunViolence", "sentence_frames")
    df_sentence_frame = pandas.read_csv(os.path.join(sentence_frame_dir, "SentenceFrames.csv"))
    id_to_sentence_frames = {}
    for i in range(df_sentence_frame.shape[0]):
        if not df_sentence_frame["ID"][i] in id_to_sentence_frames:
            id_to_sentence_frames[df_sentence_frame["ID"][i]] = [0] * 10
        id_to_sentence_frames[df_sentence_frame["ID"][i]][df_sentence_frame["Frame"][i]] += 1
        


def create_error_table(graph_dir, community_dir, output_dir):
    # get wiki mappings
    cache_dir = os.path.join(CACHE_DIR, "WikiCategoryGraph", "GunViolence")
    with open(os.path.join(cache_dir, "wiki_article_to_news_ids.json"), "r") as inputFile:
        wiki_article_to_news_ids = json.load(inputFile)
    with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "r") as inputFile:
        category_to_wiki_articles = json.load(inputFile)
    with open(os.path.join(cache_dir, "root_to_categories.json"), "r") as output_file:
        root_to_categories = json.load(output_file)
    
    
    sentence_frame_dir = os.path.join(TABLE_DIR, "NYT_Tagger", "GunViolence", "sentence_frames")
    df_sentence_frames = pandas.read_csv(os.path.join(sentence_frame_dir, "SentenceFrames.csv"))
    ids = df_sentence_frames["ID"].values.tolist()
    # news id to sentence frames
    id_to_sentence_frame_counts = {}
    for i in range(df_sentence_frames.shape[0]):
        id = df_sentence_frames["ID"][i]
        if not id in id_to_sentence_frame_counts:
            id_to_sentence_frame_counts[id] = np.zeros(10)
        id_to_sentence_frame_counts[id][df_sentence_frames["Frame"][i]] += 1

    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    
    # choose the best outcome
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Coefficient" : 1,
                  "FE Existence Similarity Function" : "cosine",
                  "FE Semantics Coefficient" : 1,
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    
    selected_graph = None
    for graph in graphs:
        select = True
        for key in conditions:
            if conditions[key] != graph[key]:
                select = False
                break
        if select:
            selected_graph = graph
    
    # get FE labels
    df_parameters = pandas.read_csv(os.path.join(community_dir, "all_parameter_values.csv"))
    table_index = None
    for i in range(df_parameters.shape[1] - 1):
        parameters = {df_parameters["Parameter"][j] : df_parameters["Table File " + str(i)][j] for j in range(df_parameters.shape[0])}
        select = True
        for key in conditions:
            if str(conditions[key]) != parameters[key]:
                select = False
                break
        if select:
            table_index = i
            break
    community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels_" + str(table_index) + ".xlsx"))
    fig_dir = os.path.join(FIGURE_DIR, "WikiCategoryGraph", "GunViolence", "evaluation")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    
    writer = pandas.ExcelWriter(os.path.join(output_dir, "clusters_to_frames_errors.xlsx"), engine = "openpyxl")
    for n in range(2, 21):
        for cm in ["SC", "VEC"]:
            df_labels = pandas.read_excel(community_excel_file, sheet_name = cm + "_" + str(n) + "_comms")
            fe_to_labels = {df_labels["Framing Element"][i] : df_labels["Label"][i] for i in range(df_labels.shape[0])}
            
            
            
            # news id to cluster labels
            id_to_community_counts = {id : np.zeros(n) for id in ids}
            for fe in fe_to_labels:
                categories = root_to_categories[fe]
                wiki_articles = []
                for category in categories:
                    wiki_articles += category_to_wiki_articles[category]
                wiki_articles = list(set(wiki_articles))
                for article in wiki_articles:
                    for id in wiki_article_to_news_ids[article]:
                        if id in id_to_community_counts:
                            id_to_community_counts[id][fe_to_labels[fe]] += 1
        
            
            
            # how many unique communities are attached to each article?
            community_counts = [np.sum(id_to_community_counts[id] > 0) for id in id_to_community_counts]
            community_counts_hitogram = np.zeros(n)
            for i in range(len(community_counts_hitogram)):
                community_counts_hitogram[i] = community_counts.count(i)
            fig, ax = plt.subplots(figsize = (12, 10))
            ax.bar([i for i in range(len(community_counts_hitogram))], community_counts_hitogram)
            ax.set_xticks([i for i in range(len(community_counts_hitogram))])
            ax.set_xticklabels([i for i in range(len(community_counts_hitogram))])
            ax.set_title("Histogram of Number of Communities Attached to News Articles", fontsize = 16)
            ax.grid(axis = "y")
            ax.set_xlabel("Number of Communities", fontsize = 14)
            ax.set_ylabel("Number of News Articles", fontsize = 14)
            fig.savefig(os.path.join(fig_dir, "CommunityNumbersAttachedtoNewsArticles_" + cm + "_" + str(n) + "_comms.png"))
            plt.close(fig)
            
            frame_adjusting_methods = [None, "average", "custom"]
            embedding_methods = ["BERT", "Doc2Vec Wiki", "Doc2Vec KaggleNews"]
            mapping_methods = ["centroid", "voting"]
            method_tuples = list(itertools.product(frame_adjusting_methods, embedding_methods, mapping_methods))
            mapping_excel_file = pandas.ExcelFile(os.path.join(output_dir, "clusters_to_frames_" + cm + "_" + str(n) + "_comms.xlsx"))
            df_index = pandas.read_excel(mapping_excel_file, sheet_name = "Index")
            errors = {"1 Norm" : np.zeros(df_index.shape[0]), "Jensen-Shannon" : np.zeros(df_index.shape[0])}
            for i in range(df_index.shape[0]):
                df_mapping = pandas.read_excel(mapping_excel_file, sheet_name = str(i))
                
                frames = [f.lower() for f in GV_FRAMES]
                community_frame_scores = []
                for j in range(df_mapping.shape[0]):
                    scores = [0.0] * 9
                    for rank in range(1, 10):
                        scores[frames.index(df_mapping["Frame " + str(rank)][j])] = df_mapping["Frame " + str(rank) + " score"][j]
                    community_frame_scores.append(scores)
                community_frame_scores = np.array(community_frame_scores)
                
                
                
                # id to frame distribution, according to community detection
                id_to_community_frame_distribution = {}
                for id in id_to_community_counts:
                    distribution = [np.zeros(9)]
                    for j, count in enumerate(id_to_community_counts[id]):
                        if count > 0:
                            distribution.append(community_frame_scores[j])
                    distribution = np.sum(np.vstack(distribution), axis = 0)
                    if (distribution > 0).any():
                        id_to_community_frame_distribution[id] = distribution / np.sum(distribution)
                        
                # id to sentence frame distribution
                id_to_sentence_frame_distribution = {}
                for id in id_to_sentence_frame_counts:
                    if (id_to_sentence_frame_counts[id][1:] > 0).any():
                        id_to_sentence_frame_distribution[id] = id_to_sentence_frame_counts[id][1:] / np.sum(id_to_sentence_frame_counts[id][1:])
                
                average_error_sum = 0
                average_JS_sum = 0
                comparable_ids = [id for id in id_to_community_frame_distribution if id in id_to_sentence_frame_distribution]
                for id in comparable_ids:
                    average_error_sum += np.sum(np.abs(id_to_community_frame_distribution[id] - id_to_sentence_frame_distribution[id]))
                    average_JS_sum += sp.spatial.distance.jensenshannon(id_to_community_frame_distribution[id], id_to_sentence_frame_distribution[id])
                average_error_sum /= len(comparable_ids)
                average_JS_sum /= len(comparable_ids)
                errors["1 Norm"][i] = average_error_sum
                errors["Jensen-Shannon"][i] = average_JS_sum
            df_errors = df_index.copy()
            for key in errors:
                df_errors[key] = errors[key]
            df_errors.to_excel(writer, sheet_name = cm + "_" + str(n) + "_comms")
    writer.close()

def create_error_plots(graph_dir, community_dir, output_dir):
    fig_dir = os.path.join(FIGURE_DIR, "WikiCategoryGraph", "GunViolence", "evaluation")
    excel_file = pandas.ExcelFile(os.path.join(output_dir, "clusters_to_frames_errors.xlsx"))
    target_index = 0 # average, Doc2Vec, centroid
    errors = {"1 Norm" : {"SC" : [], "VEC" : []}, "Jensen-Shannon" : {"SC" : [], "VEC" : []}}
    clustering_methods = ["SC", "VEC"]
    n_communities = [i for i in range(2, 21)]
    for cm in clustering_methods:
        for n in n_communities:
            df_index = pandas.read_excel(excel_file, sheet_name = cm + "_" + str(n) + "_comms")
            for key in errors:
                errors[key][cm].append(df_index[key][target_index])
    
    suffix = str(df_index["Frame adjusting method"][target_index]) + "_" + str(df_index["Embedding method"][target_index]) + "_" + str(df_index["Mapping method"][target_index])
    for key in errors:
        fig, ax = plt.subplots(figsize = (8, 6))
        for cm in errors[key]:
            ax.plot(errors[key][cm], "-o", label = cm)
        ax.grid(True)
        ax.set_xlabel("Number of Communities", fontsize = 16)
        ax.set_ylabel(key, fontsize = 16)
        ax.set_xticks([i for i in range(len(n_communities))])
        ax.set_xticklabels(n_communities)
        ax.legend(prop = {"size" : 16})
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "errors_" + key + "_" + suffix + ".png"))
        plt.close(fig)

def evaluate_results(graph_dir, community_dir, output_dir):
    # create_error_table(graph_dir, community_dir, output_dir)
    create_error_plots(graph_dir, community_dir, output_dir)

def plot_wiki_to_news_similarity_trends():
    cache_dir = os.path.join(CACHE_DIR, "WikiCategoryGraph", "GunViolence")
    scores = np.load(os.path.join(cache_dir, "SimilarityScores.npy"))
    print(scores.shape)
    histogram = np.zeros((100, 4))
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            histogram[j, int(scores[i, j] * 10) - 4] += 1
    fig_dir = os.path.join(FIGURE_DIR, "WikiCategoryGraph", "GunViolence", "4Levels_CategoryGraphs_BottomUp")
    fig, ax = plt.subplots(figsize = (40, 10))
    sns.heatmap(histogram, ax = ax)
    ax.set_xlabel("Cosine Similarity", fontsize = 14)

def plot_label_counts_stats():
    excel_file = pandas.ExcelFile(os.path.join(TABLE_DIR, "community_detection_outputs", "WikiCategoryGraph", "GunViolence", "community_labels_8.xlsx"))
    fig_dir = os.path.join(FIGURE_DIR, "WikiCategoryGraph", "GunViolence", "label counts")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    
    max_counts = [[], []]
    for i in range(2, 21):
        df_SC = pandas.read_excel(excel_file, sheet_name = "SC_" + str(i) + "_comms")
        df_VEC = pandas.read_excel(excel_file, sheet_name = "VEC_" + str(i) + "_comms")
        labels = [df_SC["Label"].values.tolist(), df_VEC["Label"].values.tolist()]
        counts=  [[labels[0].count(j) for j in range(i)], [labels[1].count(j) for j in range(i)]]
        max_counts[0].append(max(counts[0]))
        max_counts[1].append(max(counts[1]))
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.bar(np.array([j for j in range(i)]) - 0.2, counts[0], 0.4, label = "SC")
        ax.bar(np.array([j for j in range(i)]) + 0.2, counts[1], 0.4, label = "VEC")
        ax.set_xlabel("Community Label", fontsize = 16)
        ax.set_ylabel("Count", fontsize = 16)
        ax.set_xticks([j for j in range(i)])
        ax.set_xticklabels([j for j in range(i)])
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))
        ax.legend(prop = {"size" : 16})
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, str(i) + "_comms.png"))
        plt.close(fig)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot([i for i in range(2, 21)], max_counts[0], "-o", label = "SC")
    ax.plot([i for i in range(2, 21)], max_counts[1], "-o", label = "VEC")
    ax.set_xlabel("Community Number", fontsize = 16)
    ax.set_ylabel("Greatest Community Size", fontsize = 16)
    ax.set_xticks([i for i in range(2, 21)])
    ax.set_xticklabels([i for i in range(2, 21)])
    
    ax.legend(prop = {"size" : 18})
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "GreatestCommunitySizes.png"))
    plt.close(fig)

def get_stopwords_categories():
    cache_dir = os.path.join(CACHE_DIR, "WikiCategoryGraph", "GunViolence")
    with open(os.path.join(cache_dir, "root_to_categories.json"), "r") as f:
        root_to_categories = json.load(f)
    stopwords = ["wikiproject", "maintenance", "article", "branch",
                "position", "categorization", "terminology", "description",
                "namespace", "sorting", "administration", "page",
                "stub", "list", "taxonomy", "naming",
                "identifier", "cartography", "geocode", "encoding",
                "notation", "type", "topic", "category",
                "position", "navigation", "earth", "ending", "beginning"]
    with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "r") as f:
        category_to_wiki_articles = json.load(f)
    
    all_root_words = [*root_to_categories]
    tf = np.zeros(len(all_root_words))
    idf = np.zeros(len(all_root_words))
    for i in range(len(all_root_words)):
        root = all_root_words[i]
        tf[i] = len(root_to_categories[root])
        all_articles = []
        for category in root_to_categories[root]:
            all_articles += category_to_wiki_articles[category]
        idf[i] = len(set(all_articles))
    tfidf = tf / idf
    tuples = [(all_root_words[i], tfidf[i]) for i in range(len(all_root_words))]
    tuples.sort(reverse = True, key = lambda x : x[1])
    sorted_root_words = [t[0] for t in tuples]
    print(sorted_root_words[:20])
    print(sorted_root_words[20 : 40])
    print(sorted_root_words[40 : 60])
    print(sorted_root_words[60 : 80])
    print(sorted_root_words[80 : 100])

if __name__ == "__main__":
    families = ["pair counting", "graph-aware"]
    approach = "WikiCategoryGraph"
    news = "GunViolence"
    K = 100
    rank_condition = "Pages"
    
    graph_dir = os.path.join(GRAPH_DIR, news)
    '''
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs = json.load(f)
    '''
            
    
    # compute_labeling_similarity_between_semantics_weights(families, approach, news, K, rank_condition, graphs)
    
    community_dir = os.path.join(TABLE_DIR, "community_detection_outputs", approach, news)
    output_dir = os.path.join(TABLE_DIR, "label_to_frame_mappings", approach, news)
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    # map_clusters_to_frames(graph_dir, community_dir, output_dir)
    
    # compare_Joyce_labels_with_sentence_frames(graph_dir, community_dir, output_dir)
    
    evaluate_results(graph_dir, community_dir, output_dir)
    # plot_wiki_to_news_similarity_trends()
    # plot_label_counts_stats()
    
    # get_stopwords_categories()