from common import *
from WikiCategoryGraph import *
from itertools import combinations
from pytrie import StringTrie
import gensim
import stanza
import inflect
import torch
from transformers import BertModel, BertConfig, BertTokenizer
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.sep, "scratch", "lais823", "HuggingfaceTransformersCache")

###############################################################################
############################## Start of Block #################################
"""
The functions in this block fetches categories from Wikipedia. Currently it
fetches from the online database. Hence, everytime the code is executed, the
results may be different.

We are working on the alternative version that reads from an offline database
for the sake of reproducibility. However, the drawback is that the user must
install the database on their machine and this process is extremely time consuming.

Note: right now the wikipediaapi package used here cannot fetch a number of
Wiki page JSON object correctly and the reason is unknown. This is likely due
to server side issue. It gives another reason to use the offline database.
"""

def find_top_k_wiki_articles(model_path, news_dir, page_nodes_dir, top_k = 10, max_attempts = 100):
    print("Finding top " + str(top_k) + " most similar Wiki articles for each news article.")
    model = Doc2Vec.load(model_path)
    # print(str(model))
    news_vectors = None
    with h5py.File(os.path.join(news_dir, "GensimDoc2VecPV-DBOWNewsVectors.h5"), "r") as h5f:
        news_vectors = h5f["NewsVectors"][()]
    wiki = wikipediaapi.Wikipedia(language = 'en', extract_format = wikipediaapi.ExtractFormat.WIKI)
    page_nodes = []
    similarity_scores = []
    errors = []
    
    
    for i in trange(len(news_vectors)):
        example = news_vectors[i]
        top_matches = model.dv.most_similar([example], topn = top_k)
        titles = [tuple[0] for tuple in top_matches]
        similarity_scores += [tuple[1] for tuple in top_matches]
        for j in range(len(titles)):
            title = titles[j]
            attempts, success = 0, False
            while attempts < max_attempts:
                try:
                    page = wiki.page(title)
                    pageID = page.pageid
                    new_node = GraphNode(title, pageID, "page")
                    for category in page.categories:
                        new_node.add_parent(category)
                    page_nodes.append(new_node)
                    attempts = max_attempts
                    success = True
                except:
                    attempts += 1
            if not success:
                errors.append("Unable to retrieve title " + str(j) + " [" + title + "] for example " + str(i) + ".")
            if j > 0 and j % 100 == 0:
                with open(os.path.join(page_nodes_dir, "PageNodes.pkl"), "wb") as output_file:
                    pickle.dump(page_nodes, output_file, pickle.DEFAULT_PROTOCOL)
                similarity_scores = np.array(similarity_scores)
                np.save(os.path.join(page_nodes_dir, "SimilarityScores.npy"), similarity_scores)
                with open(os.path.join(page_nodes_dir, "Error.log"), "w") as f:
                    f.write("\n".join(errors))
    
    '''
    for i in trange(len(news_vectors)):
        example = news_vectors[i]
        top_matches = model.dv.most_similar([example], topn = top_k)
        titles = [tuple[0] for tuple in top_matches]
        similarity_scores += [tuple[1] for tuple in top_matches]
        for j in range(len(titles)):
            try:
                title = titles[j]
                page = wiki.page(title)
                pageID = page.pageid
                new_node = GraphNode(title, pageID, "page")
                for category in page.categories:
                    new_node.add_parent(category)
                page_nodes.append(new_node)
                attempts = max_attempts
                success = True
            except:
                title = titles[j]
                page = wiki.page(title)
                print(title)
                print(page)
                sys.exit(1)
    '''
    with open(os.path.join(page_nodes_dir, "PageNodes.pkl"), "wb") as output_file:
        pickle.dump(page_nodes, output_file, pickle.DEFAULT_PROTOCOL)
    similarity_scores = np.array(similarity_scores)
    np.save(os.path.join(page_nodes_dir, "SimilarityScores.npy"), similarity_scores)
    with open(os.path.join(page_nodes_dir, "Error.log"), "w") as f:
        f.write("\n".join(errors))

def trace_categories(wiki, category_record, error_record, current, category_to_pages,
                        original_page, category_count, level, max_level, max_attempts = 100):
    if level <= max_level:
        attempt, done = 0, False
        while attempt < max_attempts and not done:
            try:
                wiki_page = wiki.page(current)
                categories = wiki_page.categories
                for category in categories:
                    if is_helpful_category(wiki, category):
                        if not category in category_to_pages:
                            category_to_pages[category] = [(original_page, level)]
                        else:
                            category_to_pages[category].append((original_page, level))
                        
                        if not category in category_record[level]:
                            category_record[level][category] = 1
                        else:
                            category_record[level][category] += 1
                        
                        category_count += 1
                        trace_categories(wiki, category_record, error_record,
                                            category, category_to_pages,
                                            original_page, category_count, level + 1, max_level)
                        '''
                        if category_count % 100 == 0:
                            print(str(category_count) + " categories explored.")
                        '''
                done = True
            except:
                print(sys.exc_info()[0])
                attempt += 1
        if attempt == max_attempts and not done:
            error_record.append((current, original_page, level, max_level))
            # print("Error occurred at " + str(current) + " (level " + str(level) + ").")

def make_category_graphs_multi_level(page_nodes_dir, output_dir, levels, batch_size = None, batch_index = None):
    # load page nodes
    page_nodes = None
    with open(os.path.join(page_nodes_dir, "PageNodes.pkl"), "rb") as inputFile:
        page_nodes = pickle.load(inputFile)
    page_names = list(set([node.name for node in page_nodes]))

    start = 0
    end = len(page_names)
    if not batch_size is None and not batch_index is None:
        start = batch_size * batch_index
        end = min(end, start + batch_size)
    if start >= end:
        print("Invalid start(" + str(start) + ") and end(" + str(end) + ") number.")
        sys.exit(1)
    print("Batch size = " + str(batch_size))
    print("Current batch: " + str(batch_index) + ".")
    wiki = wikipediaapi.Wikipedia(language = 'en', extract_format = wikipediaapi.ExtractFormat.WIKI)
    category_record = {i : {} for i in range(levels + 1)}
    category_to_pages = {}
    error_record = []
    for i in range(start, end):
        if i - start > 0 and (i - start) % 100 == 0:
            print("100 pages explored.")
        trace_categories(wiki, category_record, error_record, page_names[i], category_to_pages, page_names[i], 0, 1, levels)
    for category in category_to_pages:
        category_to_pages[category] = list(set(category_to_pages[category]))
    # pprint(category_to_pages)
    output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryRecord"
    if not batch_size is None and not batch_index is None:
        output_filename += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename + ".json"), "w") as output_file:
        output_file.write(json.dumps(category_record))
    
    output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryToPages"
    if not batch_size is None and not batch_index is None:
        output_filename += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename + ".json"), "w") as output_file:
        output_file.write(json.dumps(category_to_pages))
    
    output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_ErrorRecord"
    if not batch_size is None and not batch_index is None:
        output_filename += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename + ".pkl"), "wb") as f:
        pickle.dump(error_record, f, pickle.DEFAULT_PROTOCOL)
    
def make_category_graphs_multi_level_error_fixing(page_nodes_dir, output_dir, levels, batch_size = None, batch_index = None):
    print("Retrying unfinished jobs in batch " + str(batch_index) + ".")
    output_filename_category_record = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryRecord"
    if not batch_size is None and not batch_index is None:
        output_filename_category_record += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename_category_record + ".json"), "r") as output_file:
        category_record = json.load(output_file)
    category_record = {level : category_record[str(level)] for level in range(levels + 1)}
    # print("Category record:")
    # pprint(category_record)
    
    output_filename_category_to_pages = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryToPages"
    if not batch_size is None and not batch_index is None:
        output_filename_category_to_pages += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename_category_to_pages + ".json"), "r") as output_file:
        category_to_pages = json.load(output_file)
    # print("category_to_pages:")
    # pprint(category_to_pages)
    
    error_record = None
    output_filename_error_record = str(levels) + "Levels_CategoryGraphs_BottomUp_ErrorRecord"
    if not batch_size is None and not batch_index is None:
        output_filename_error_record += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename_error_record + ".pkl"), "rb") as f:
        error_record = pickle.load(f)
    print("Error record: " + str(error_record))
    
    
    wiki = wikipediaapi.Wikipedia(language = 'en', extract_format = wikipediaapi.ExtractFormat.WIKI)
    error_record_new = []
    for tuple in error_record:
        current_category, original_page, level, levels = tuple[0], tuple[1], tuple[2], tuple[3]
        trace_categories(wiki, category_record, error_record_new, current_category, category_to_pages, original_page, 0, level, levels)
    print("New error record: " + str(error_record_new))
    
    print("Saving.")
    with open(os.path.join(output_dir, output_filename_category_record + ".json"), "w") as output_file:
        output_file.write(json.dumps(category_record))
    with open(os.path.join(output_dir, output_filename_category_to_pages + ".json"), "w") as output_file:
        output_file.write(json.dumps(category_to_pages))
    with open(os.path.join(output_dir, output_filename_error_record + ".pkl"), "wb") as f:
        pickle.dump(error_record_new, f, pickle.DEFAULT_PROTOCOL)
    
    print("Complete!\n")


def check_multi_level_category_graph_output(page_nodes_dir, output_dir, levels, batch_size = None, batch_index = None):
    print("Batch " + str(batch_index))
    output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryRecord"
    if not batch_size is None and not batch_index is None:
        output_filename += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename + ".json"), "r") as output_file:
        category_record = json.load(output_file)
    # print("Category record:")
    # pprint(category_record)
    
    output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryToPages"
    if not batch_size is None and not batch_index is None:
        output_filename += "_Batch" + str(batch_index)
    with open(os.path.join(output_dir, output_filename + ".json"), "r") as output_file:
        category_to_pages = json.load(output_file)
    # print("category_to_pages:")
    # pprint(category_to_pages)
    
    error_record = None
    with open(os.path.join(output_dir, output_filename + "_error_record.pkl"), "rb") as f:
        error_record = pickle.load(f)
    print("Error record:")
    print(error_record)

############################### End of Block ##################################
###############################################################################

def find_nonhidden_categories(categories, max_attempts = 100):
    session = requests.Session()
    nonhidden_categories = []
    for i in trange(len(categories)):
        category = categories[i]
        attempts, is_hidden = 0, None
        while attempts < max_attempts and is_hidden is None:
            try:
                params = {
                    "action": "query",
                    "titles": category,
                    "prop": "pageprops",
                    "format": "json"
                }
                response = session.get(url = MEDIAWIKI_URL, params = params)
                data = response.json()
                is_hidden = "hiddencat" in data["query"]["pages"][next(iter(data["query"]["pages"]))]["pageprops"]
            except:
                attempts += 1
        if is_hidden is None:
            print("Cannot retrieve info for " + str(category))
        elif not is_hidden:
            nonhidden_categories.append(category)
    return nonhidden_categories

def get_Wiki_to_news_article_mappings(page_nodes_dir, download_dir, news_dir, cache_dir, levels, batch_size):
    wiki_article_to_news_ids, category_to_wiki_articles = None, None
    try:
        with open(os.path.join(cache_dir, "wiki_article_to_news_ids.json"), "r") as f:
            wiki_article_to_news_ids = json.load(f)
        with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "r") as f:
            category_to_wiki_articles = json.load(f)
    except:
        # load page nodes
        print("Loading page nodes.")
        page_nodes = None
        with open(os.path.join(page_nodes_dir, "PageNodes.pkl"), "rb") as inputFile:
            page_nodes = pickle.load(inputFile)
        page_names = list(set([node.name for node in page_nodes]))
        
        # load news articles
        df_news = pandas.read_csv(os.path.join(news_dir, "final_gv_fulltext_url.csv"))
        
        # build page-to-article_id mapping
        print("Building Wiki article to news article mapping.")
        wiki_article_to_news_ids = {name : [] for name in page_names}
        k = 10 # the number of pages per article
        for i in range(df_news.shape[0]):
            for j in range(i * k, (i + 1) * k):
                wiki_article_to_news_ids[page_nodes[j].name].append(df_news["ID"][i])
        
        # load category-to-page mappings
        print("Loading category-to-page mappings.")
        output_filename = str(levels) + "Levels_CategoryGraphs_BottomUp_CategoryToPages_Batch"
        max_batch_index = len(page_names) // batch_size
        
        category_to_wiki_articles = {}
        for i in range(max_batch_index):
            with open(os.path.join(download_dir, output_filename + str(i) + ".json"), "r") as output_file:
                category_to_wiki_articles_batch = json.load(output_file)
            for category in category_to_wiki_articles_batch:
                if not category in category_to_wiki_articles:
                    category_to_wiki_articles[category] = []
                    for item in category_to_wiki_articles_batch[category]:
                        category_to_wiki_articles[category].append(item[0])
        
        print("Removing duplicates in category-to-page mappings.")
        for category in category_to_wiki_articles:
            category_to_wiki_articles[category] = list(set(category_to_wiki_articles[category]))
        print("Total number of unique categories: " + str(len(category_to_wiki_articles)))
        with open(os.path.join(cache_dir, "wiki_article_to_news_ids.json"), "w") as output_file:
            output_file.write(json.dumps(wiki_article_to_news_ids))
        with open(os.path.join(cache_dir, "category_to_wiki_articles.json"), "w") as output_file:
            output_file.write(json.dumps(category_to_wiki_articles))
    return wiki_article_to_news_ids, category_to_wiki_articles

def clean_categories(category_to_pages, cache_dir):
    print("Identifying hidden categories.")
    nonhidden_categories = []
    try:
        with open(os.path.join(cache_dir, "nonhidden_categories.pkl"), "rb") as f:
            nonhidden_categories = pickle.load(f)
    except:
        nonhidden_categories = find_nonhidden_categories([*category_to_pages])
        with open(os.path.join(cache_dir, "nonhidden_categories.pkl"), "wb") as f:
            pickle.dump(nonhidden_categories, f, pickle.DEFAULT_PROTOCOL)
            
    
    print("Removing hidden categories.")
    temp = {}
    for i in range(len(nonhidden_categories)):
        temp[nonhidden_categories[i]] = category_to_pages[nonhidden_categories[i]]
    category_to_pages = temp
    print("Total number of unique nonhidden categories: " + str(len(category_to_pages)))
    
    print("Removing Wikipedia system categories.")
    keywords = ["wikiproject", "stub", "wikipedia", "pages with"]
    temp = {}
    for category in category_to_pages:
        category_lower = category.lower()
        is_bad_category = False
        for w in keywords:
            if w in category_lower:
                is_bad_category = True
                break
        if not is_bad_category:
            temp[category] = category_to_pages[category]
    category_to_pages = temp
    print("Total number of categories left: " + str(len(category_to_pages)))
    
    # use corenlp to keep only the root
    print("Grouping categories by dependency parsing.")
    root_to_categories = {}
    
    try:
        with open(os.path.join(cache_dir, "root_to_categories.json"), "r") as output_file:
            root_to_categories = json.load(output_file)
    except:
        nlp = stanza.Pipeline(lang = "en", dir = STANZA_DIR, processors = "tokenize,mwt,pos,lemma,depparse", verbose = False)
        p = inflect.engine()
        categories = [*category_to_pages]
        
        for i in trange(len(categories)):
            category = categories[i]
            doc = nlp(category[len("Category:"):].lower())
            sentence = doc.sentences[0]
            for word in sentence.words:
                if word.deprel == "root":
                    root = p.singular_noun(word.text)
                    if not root:
                        root = word.text
                    if not root.isnumeric():
                        if not root in root_to_categories:
                            root_to_categories[root] = []
                        root_to_categories[root].append(category)
        # save root_to_categories
        with open(os.path.join(cache_dir, "root_to_categories.json"), "w") as output_file:
            output_file.write(json.dumps(root_to_categories))
    '''
    # bad categories picked by Joyce
    to_drop = ["wikiproject", "maintenance", "article", "branch",
                "position", "categorization", "terminology", "description",
                "namespace", "sorting", "administration", "page",
                "stub", "list", "taxonomy", "naming",
                "identifier", "cartography", "geocode", "encoding",
                "notation", "type", "topic", "category",
                "position", "navigation", "earth", "ending", "beginning"]
    root_to_categories = {root : root_to_categories[root] for root in root_to_categories if not root in to_drop}
    '''
    print("Number of roots: " + str(len(root_to_categories)) + ".")
    return root_to_categories

if __name__ == "__main__":
    start_time = time.time()
    page_nodes_dir = os.path.join(CACHE_DIR, "wiki_category_graph", "GunViolence", "PageNodes")
    pathlib.Path(page_nodes_dir).mkdir(parents = True, exist_ok = True)
    model_path = os.path.join(GENSIM_DIR, "GensimModels", "Doc2Vec " + str(gensim.__version__), "Wiki-" + DUMPS_TIME, DOC2VEC_MODEL_TYPE, "EN_Wiki_Pages_Articles.model")
    news_dir = os.path.join(DATA_DIR, "News", "GunViolence")
    
    levels = 1
    wiki_download_dir = os.path.join(CACHE_DIR, "wiki_category_graph", "GunViolence", "downloads", str(levels) + "Levels_CategoryGraphs_BottomUp")
    pathlib.Path(wiki_download_dir).mkdir(parents = True, exist_ok = True)
    
    if len(sys.argv) > 1:
        if "-f" in sys.argv or "-find" in sys.argv:
            top_k = 10
            if "-k" in sys.argv:
                top_k = int(sys.argv[sys.argv.index("-k") + 1])
            find_top_k_wiki_articles(model_path, news_dir, page_nodes_dir, top_k)
        if "-e" in sys.argv or "-explore" in sys.argv:
            output_dir = os.path.join(CACHE_DIR, "WikiCategoryGraph", "GunViolence", str(levels) + "Levels_CategoryGraphs_BottomUp", "downloads")
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            batch_size, batch_index = None, None
            if "-batch_size" in sys.argv:
                batch_size = int(sys.argv[sys.argv.index("-batch_size") + 1])
            if "-batch_index" in sys.argv:
                batch_index = int(sys.argv[sys.argv.index("-batch_index") + 1])
            make_category_graphs_multi_level(page_nodes_dir, wiki_download_dir, levels, batch_size, batch_index)
        if "-c" in sys.argv or "-clean" in sys.argv:
            levels, batch_size = 4, 250
            cache_dir = os.path.join(CACHE_DIR, "wiki_category_graph", "GunViolence")
            wiki_article_to_news_ids, category_to_wiki_articles = get_Wiki_to_news_article_mappings(page_nodes_dir, wiki_download_dir, news_dir, cache_dir, levels, batch_size)
            root_to_categories = clean_categories(category_to_wiki_articles, cache_dir)
    end_time = time.time()
    time_elapsed = np.round((end_time - start_time) / 60, 2)
    print("Time elapsed = " + str(time_elapsed) + " minutes.")