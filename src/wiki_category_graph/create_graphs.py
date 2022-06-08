from common import *
from WikiCategoryGraph import *
from itertools import combinations
from pytrie import StringTrie
import stanza
import inflect
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.sep, "scratch", "lais823", "HuggingfaceTransformersCache")

def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

def jaccard(x, y):
    x = np.asarray(x, np.bool) # Not necessary, if you keep your data
    y = np.asarray(y, np.bool) # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

def get_BERT_embeddings(texts):
    MODEL_TYPE = "bert-base-uncased"
    MAX_SIZE = 150
    device = torch.device("cuda")
    
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
    model = BertModel.from_pretrained(MODEL_TYPE)
    model.to(device)
    
    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    
    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
    dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)
    embeddings = []
    for step, batch in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
        embeddings.append(outputs[0][:,0,:].detach().cpu().numpy().squeeze())
    return embeddings

def make_adjacency_matrix(existence_vectors, existence_similarity_function, BERT_embeddings, c_ex, c_sem):
    elements = [*existence_vectors]
    n = len(elements)
    M_Ex = np.zeros((n, n), dtype = "double")
    M_Sem = np.zeros((n, n), dtype = "double")
    for i in range(n):
        e1 = elements[i]
        for j in range(i + 1, n):
            e2 = elements[j]
            if c_ex:
                M_Ex[i, j] = existence_similarity_function(existence_vectors[e1], existence_vectors[e2])
            if c_sem:
                M_Sem[i, j] = sp.spatial.distance.cosine(BERT_embeddings[e1], BERT_embeddings[e2])
    M = c_ex * M_Ex + c_sem * M_Sem
    M += M.T
    return M

def create_graphs_for_community_detection(n_roots_values, graph_dir, cache_dir):
    wiki_article_to_news_ids, category_to_wiki_articles, root_to_categories = None, None, None
    
    with open(os.path.join(cache_dir, "wiki_article_to_news_ids.json"), "r") as f:
        wiki_article_to_news_ids = json.load(f)
    with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "r") as f:
        category_to_wiki_articles = json.load(f)
    with open(os.path.join(cache_dir, "root_to_categories.json"), "r") as f:
        root_to_categories = json.load(f)
    
    
    all_wiki_articles = []
    for category in category_to_wiki_articles:
        all_wiki_articles += category_to_wiki_articles[category]
    all_wiki_articles = list(set(all_wiki_articles))
    
    all_news_ids = []
    for page in all_wiki_articles:
        all_news_ids += wiki_article_to_news_ids[page]
    all_news_ids = list(set(all_news_ids))
    
    print("Building existence vectors.")
    existence_vectors = {"Wiki articles" : {}, "News articles" : {}}
    for root in root_to_categories:
        wiki_articles = []
        for category in root_to_categories[root]:
            wiki_articles += category_to_wiki_articles[category]
        wiki_articles = set(wiki_articles)
        vector = np.zeros(len(all_wiki_articles))
        for i in range(len(all_wiki_articles)):
            if all_wiki_articles[i] in wiki_articles:
                vector[i] = 1
        existence_vectors["Wiki articles"][root] = vector
        
        news_ids = []
        for wiki_article in wiki_articles:
            news_ids += wiki_article_to_news_ids[wiki_article]
        news_ids = set(news_ids)
        vector = np.zeros(len(all_news_ids))
        for i in range(len(all_news_ids)):
            if all_news_ids[i] in news_ids:
                vector[i] = 1
        existence_vectors["News articles"][root] = vector
    
    print("Building semantics vectors.")
    fe_semantics_method = "BERT"
    all_roots = [*root_to_categories]
    bert_embeddings = get_BERT_embeddings(all_roots)
    # bert_embeddings = np.ones((len(all_roots), 10)) # for testing only
    semantics_vectors = {all_roots[i] : bert_embeddings[i] for i in range(len(all_roots))}
    print("Building graph objects.")
    '''
    # for exploration only
    ranked_by = ["Wiki articles", "News articles"]
    fe_existence_similarity_functions = [cosine, jaccard]
    fe_existence_coefficients = [0, 0.5, 1]
    fe_semantics_coefficients = [0, 0.5, 1]
    '''
    
    # for final product
    ranked_by = ["Wiki articles"]
    fe_existence_similarity_functions = [cosine]
    fe_existence_coefficients = [1]
    fe_semantics_coefficients = [1]
    
    graphs = []
    for n_roots in n_roots_values:
        for i in range(len(ranked_by)):
            tuples = [(root, existence_vectors[ranked_by[i]][root].sum()) for root in existence_vectors[ranked_by[i]]]
            tuples.sort(reverse = True, key = lambda x : x[1])
            selected_roots = [t[0] for t in tuples[:n_roots]]
            selected_semantics_vectors = {root : semantics_vectors[root].tolist() for root in selected_roots}
            for existence_base in existence_vectors:
                selected_existence_vectors = {root : existence_vectors[existence_base][root].tolist() for root in selected_roots}
                for f in fe_existence_similarity_functions:
                    for c_ex in fe_existence_coefficients:
                        for c_sem in fe_semantics_coefficients:
                            graph = {"Framing Elements" : selected_roots,
                                     "FE Existence Base" : existence_base,
                                     "FE Existence Vectors" : selected_existence_vectors,
                                     "FE Existence Coefficient" : c_ex,
                                     "FE Existence Similarity Function" : f.__name__,
                                     "FE Semantics Method" : fe_semantics_method,
                                     "FE Semantics Vectors" : selected_semantics_vectors,
                                     "FE Semantics Coefficient" : c_sem,
                                     "FE Selection Scheme" : "top " + str(n_roots) + " nonstop root words ranked by " + ranked_by[i],
                                     "Adjacency Matrix" : make_adjacency_matrix(selected_existence_vectors, f, selected_semantics_vectors, c_ex, c_sem).tolist()}
                            graphs.append(graph)
    pathlib.Path(graph_dir).mkdir(parents = True, exist_ok = True)
    with open(os.path.join(graph_dir, "graphs.json"), "w") as f:
        json.dump(graphs, f)
    

def compare_graphs(graph_dir):
    graphs = [None, None]
    with open(os.path.join(graph_dir, "graphs.json"), "r") as f:
        graphs[0] = json.load(f)
    with open(os.path.join(graph_dir, "graphs_wiki_sys_categories_removed.json"), "r") as f:
        graphs[1] = json.load(f)
    conditions = {"FE Existence Base" : "Wiki articles",
                  "FE Existence Coefficient" : 1,
                  "FE Existence Similarity Function" : "cosine",
                  "FE Semantics Coefficient" : 1,
                  "FE Selection Scheme" : "top 100 nonstop root words ranked by Wiki articles"}
    selected_graphs = [None, None]
    for i in range(len(graphs)):
        for graph in graphs[i]:
            select = True
            for key in conditions:
                if conditions[key] != graph[key]:
                    select = False
                    break
            if select:
                selected_graphs[i] = graph
    FEs = [selected_graphs[0]["Framing Elements"], selected_graphs[1]["Framing Elements"]]
    count = 0
    temp = []
    for fe in FEs[0]:
        if fe in FEs[1]:
            count += 1
        else:
            temp.append(fe)
    print(temp)
    count = 0
    temp = []
    for fe in FEs[1]:
        if fe in FEs[0]:
            count += 1
        else:
            temp.append(fe)
    print(temp)

if __name__ == "__main__":
    start_time = time.time()

    n_roots_values = [100, 150]
    graph_dir = os.path.join(GRAPH_DIR, "GunViolence")
    # print(graph_dir)
    cache_dir = os.path.join(CACHE_DIR, "wiki_category_graph", "GunViolence")
    create_graphs_for_community_detection(n_roots_values, graph_dir, cache_dir)
    # compare_graphs(graph_dir)
    
    end_time = time.time()
    time_elapsed = np.round((end_time - start_time) / 60, 2)
    print("Time elapsed = " + str(time_elapsed) + " minutes.")