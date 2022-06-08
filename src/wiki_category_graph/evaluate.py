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
    
if __name__ == "__main__":
    pass