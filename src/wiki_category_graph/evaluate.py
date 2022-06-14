from common import *
from WikiCategoryGraph import *
import networkx as nx
import community
import partition_networkx
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, mutual_info_score
import gensim
from openpyxl import load_workbook
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def get_BERT_embeddings(texts, model_type = BERT_MODEL_TYPE, show_progress = True):
    MAX_SIZE = 150
    device = torch.device("cuda")
    
    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type)
    model.to(device)
    
    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    
    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
    dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)
    embeddings = []
    if show_progress:
        for step, batch in enumerate(tqdm(dataloader)):
            torch.cuda.empty_cache()
            with torch.no_grad():
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
            embeddings.append(outputs[0][:,0,:].detach().cpu().numpy().squeeze())
    else:
        for step, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            with torch.no_grad():
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
            embeddings.append(outputs[0][:,0,:].detach().cpu().numpy().squeeze())
    return embeddings

def get_cluster_to_frame_mappings(graph, community_excel_file, sheet_name):   
    # FE and frame embeddings
    fe_to_embedding = graph["FE Semantics Vectors"]
    frame_embeddings = get_BERT_embeddings([f.lower() for f in GV_FRAMES], show_progress = False)
    frame_to_embedding = {GV_FRAMES[i] : frame_embeddings[i] for i in range(len(GV_FRAMES))}
    
    df_labels = pandas.read_excel(community_excel_file, sheet_name = sheet_name)
    fe_to_labels = {df_labels["Frame Element"][i] : df_labels["Label"][i] for i in range(df_labels.shape[0])}
    
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
    mappings = {"Label" : sorted([*label_to_fes])}
    
    for i in range(len(GV_FRAMES)):
        mappings["Frame " + str(i + 1)] = []
        mappings["Frame " + str(i + 1) + " score"] = []
    label_to_centroid = {}
    for label in label_to_fe_embeddings:
        label_to_centroid[label] = np.mean(label_to_fe_embeddings[label], axis = 0)
    
    for label in sorted([*label_to_centroid]):
        centroid = label_to_centroid[label]
        frame_to_similarity = {f : -sys.maxsize for f in GV_FRAMES}
        for frame in frame_to_embedding:
            similarity = cosine_similarity((centroid, frame_to_embedding[frame]))
            frame_to_similarity[frame] = max(frame_to_similarity[frame], similarity)
        tuples = [(frame, frame_to_similarity[frame]) for frame in frame_to_similarity]
        tuples.sort(reverse = True, key = lambda x: x[1])
        for i in range(len(tuples)):
            mappings["Frame " + str(i + 1)].append(tuples[i][0])
            mappings["Frame " + str(i + 1) + " score"].append(tuples[i][1])
    return mappings

def map_clusters_to_frames(graph, community_excel_file, output_dir):
    print("Mapping communities to frames.")
    clustering_methods = ["SC", "VEC"]
    n_communities = [i for i in range(2, 21)]
    writer = pandas.ExcelWriter(os.path.join(output_dir, "clusters_to_frames.xlsx"), engine = "openpyxl")
    for n in n_communities:
        for cm in clustering_methods:
            # print(cm + "_" + str(n) + "_comms")
            sheet_name = cm + "_" + str(n) + "_comms"
            mapping = get_cluster_to_frame_mappings(graph, community_excel_file, sheet_name)
            df_out = pandas.DataFrame(mapping)
            df_out.to_excel(writer, sheet_name = sheet_name, index = False)
    writer.close()

def create_error_table(cache_dir, sentence_frame_dir, graph, community_excel_file, output_dir):
    print("Creating error table.")
    # get wiki mappings
    with open(os.path.join(cache_dir, "wiki_article_to_news_ids.json"), "r") as inputFile:
        wiki_article_to_news_ids = json.load(inputFile)
    with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "r") as inputFile:
        category_to_wiki_articles = json.load(inputFile)
    with open(os.path.join(cache_dir, "root_to_categories.json"), "r") as output_file:
        root_to_categories = json.load(output_file)
    
    
    df_sentence_frames = pandas.read_csv(os.path.join(sentence_frame_dir, "SentenceFrames.csv"))
    ids = df_sentence_frames["ID"].values.tolist()
    # news id to sentence frames
    id_to_sentence_frame_counts = {}
    for i in range(df_sentence_frames.shape[0]):
        id = df_sentence_frames["ID"][i]
        if not id in id_to_sentence_frame_counts:
            id_to_sentence_frame_counts[id] = np.zeros(10)
        id_to_sentence_frame_counts[id][df_sentence_frames["Frame"][i]] += 1

    mapping_excel_file = pandas.ExcelFile(os.path.join(output_dir, "clusters_to_frames.xlsx"))
    M_JS = {"" : [i for i in range(2, 21)], "SC" : [], "VEC" : []}
    for n in range(2, 21):
        for cm in ["SC", "VEC"]:
            df_labels = pandas.read_excel(community_excel_file, sheet_name = cm + "_" + str(n) + "_comms")
            fe_to_labels = {df_labels["Frame Element"][i] : df_labels["Label"][i] for i in range(df_labels.shape[0])}
            
            
            
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
            
            df_mapping = pandas.read_excel(mapping_excel_file, sheet_name = cm + "_" + str(n) + "_comms")
            
            community_frame_scores = []
            for j in range(df_mapping.shape[0]):
                scores = [0.0] * 9
                for rank in range(1, 10):
                    scores[GV_FRAMES.index(df_mapping["Frame " + str(rank)][j])] = df_mapping["Frame " + str(rank) + " score"][j]
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
            M_JS[cm].append(average_JS_sum)
    df_error = pandas.DataFrame(M_JS)
    df_error.to_csv(os.path.join(output_dir, "clusters_to_frames_errors.csv"), index = False)


if __name__ == "__main__":
    news = "GunViolence"
    cache_dir = os.path.join(CACHE_DIR, "wiki_category_graph", news)
    sentence_frame_dir = os.path.join(TABLE_DIR, "sentence_frames", news)
    graph_dir = os.path.join(GRAPH_DIR, news)
    community_dir = os.path.join(TABLE_DIR, "community_detection_outputs", news)
    
    graph_file_index = 0
    with open(os.path.join(graph_dir, "graph_" + str(graph_file_index) + ".json"), "r") as f:
        graph = json.load(f)
    community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels_" + str(graph_file_index) + ".xlsx"))
    output_dir = os.path.join(TABLE_DIR, "evaluation")
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    fig_dir = os.path.join(FIGURE_DIR, "evaluation", "GunViolence")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    
    map_clusters_to_frames(graph, community_excel_file, output_dir)
    create_error_table(cache_dir, sentence_frame_dir, graph, community_excel_file, output_dir)