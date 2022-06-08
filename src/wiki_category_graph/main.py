from common import *
from WikiCategoryGraph import *
from multiprocessing import Pool

import xml.etree.ElementTree as etree
import sqlite3

def get_unhelpful_categories(categories_start, output_dir):
    categories_final = []
    error_record = []
    categories_current = categories_start
    categories_next = []
    max_level = 1
    for level in range(max_level):
        start_time = time.time()
        for category in categories_current:
            subcategories = get_subcategories(category)
            if subcategories is None:
                error_record.append(category)
            else:
                for item in subcategories:
                    subcategory = item["title"]
                    if not subcategory in categories_final:
                        categories_final.append(subcategory)
                        categories_next.append(subcategory)
        categories_current = categories_next
        categories_next = []
        with open(os.path.join(output_dir, "UnhelpfulCategories.pkl"), "wb") as f:
            pickle.dump(categories_final, f, pickle.DEFAULT_PROTOCOL)
        df_out = pandas.DataFrame({"Category" : categories_final})
        df_out.to_csv(os.path.join(output_dir, "UnhelpfulCategories.csv"), index = False)
        with open(os.path.join(output_dir, "ErrorRecord.pkl"), "wb") as f:
            pickle.dump(error_record, f, pickle.DEFAULT_PROTOCOL)
        end_time = time.time()
        time_elapsed = np.round((end_time - start_time) / 60, 2)
        print("Level " + str(level) + " finished. " + str(len(categories_final)) + " categories examined so far. " + str(len(error_record)) + " items in error record. Time elapsed = " + str(time_elapsed) + " minutes.")
    return categories_final

def strip_tag_name(tag):
    i = tag.rfind("}")
    if i != -1:
        tag = tag[i + 1:]
    return tag

def convert_pages_articles_dump_file(input_dir):
    wiki_dump_path = os.path.join(input_dir, "enwiki-latest-pages-articles.xml")
    M = {"id" : [], "title" : []}
    title, id, redirect, in_revision, ns = None, None, None, None, None
    for event, element in etree.iterparse(wiki_dump_path, events = ("start", "end")):
        tname = strip_tag_name(element.tag)
        if event == "start":
            if tname == "page":
                title = ""
                id = -1
                redirect = ""
                in_revision = False
                ns = 0
            elif tname == "revision":
                in_revision = True
        else:
            if tname == "title":
                title = element.text
            elif tname == "id" and not in_revision:
                id = int(element.text)
            elif tname == "redirect":
                redirect = element.attrib["title"]
            elif tname == "ns":
                ns = int(element.text)
            elif tname == "page":
                M["id"].append(id)
                M["title"].append(title)
            element.clear()
    df_out = pandas.DataFrame(M)
    df_out.to_csv(os.path.join(input_dir, "enwiki-latest-pages-articles-id_title.csv"), index = False)
    print(str(df_out.shape[0]) + " pages processed.")

def examine_wiki_db(input_dir):
    category = "Category:1750 in Great Britain"
    connection = sqlite3.connect(os.path.join(input_dir, "enwiki-latest-page_props.db"))
    df = pandas.read_sql_query("select pp_page, pp_value from page_props where pp_propname = 'hiddencat';", connection)
    
    # df.to_csv(os.path.join(input_dir, "enwiki-latest-page_hiddencat.csv"), index = False)
    
    cursor = connection.cursor()
    cursor.execute("select pp_page, pp_value from page_props where pp_propname = 'hiddencat'")
    results = list(cursor.fetchall())
    pprint(results[:10])
    print()
    cursor.execute("select pp_page, pp_value from page_props where pp_propname = 'displaytitle'")
    results = list(cursor.fetchall())
    pprint(results[:10])
    cursor.close()
    
    connection.close()
    
    '''
    connection = sqlite3.connect(os.path.join(input_dir, "enwiki-latest-page_hiddencat.db"))
    df.to_sql("page_hiddencat", connection, if_exists = "replace")
    connection.execute("""create table page_hiddencat as select * from page_hiddencat""")
    connecitn.close()
    '''

def temp():
    category_1 = "Category:CatAutoTOC generates standard Category TOC"
    category_2 = "Category:Graph theory"
    
    session = requests.Session()

    
    params = {
        "action": "query",
        "titles": category_1,
        "prop": "pageprops",
        "format": "json"
    }
    response = session.get(url = MEDIAWIKI_URL, params = params)
    data = response.json()
    print(data)
    results = data["query"]["pages"][next(iter(data["query"]["pages"]))]["pageprops"]
    print("hiddencat" in results)
    


def make_page_article_similarity_distribution_plots(processes):
    with h5py.File(os.path.join(DATA_DIR, "Wiki", "Pages", "GensimDoc2VecPV-DBOWPageVectors.h5"), "r") as h5f:
        page_article_vectors = h5f["PageVectors"][()]
    with h5py.File(os.path.join(DATA_DIR, "News", "GunViolence", "GensimDoc2VecPV-DBOWNewsVectors.h5"), "r") as h5f:
        news_article_vectors = h5f["NewsVectors"][()]
    distances = []
    for i in range(len(news_article_vectors)):
        pairs = []
        for j in range(len(page_article_vectors)):
            pairs.append((news_article_vectors[i], page_article_vectors[j]))
        pool = Pool(processes)
        L = pool.map(cosine_similarity, pairs)
        pool.close()
        distances.append(heapq.nlargest(100, L))
        if i > 0 and i % 100 == 0:
            print("i = " + str(i))
            np.save(os.path.join(CACHE_DIR, "SimilarityScores.npy"), np.array(distances))
    np.save(os.path.join(CACHE_DIR, "SimilarityScores.npy"), np.array(distances))

if __name__ == "__main__":
    start_time = time.time()
    '''
    target_categories = ["Category:Tracking categories", "Category:Hidden categories", "Category:CatAutoTOC tracking categories"]
    output_dir = os.path.join(CACHE_DIR, "GunViolence", "UnhelpfulWikiCategories")
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    unhelpful_categories = get_unhelpful_categories(target_categories, output_dir)
    '''
    wiki_dir = os.path.join(WIKI_DATA_DIR, "Dumps", DUMPS_TIME)
    # convert_pages_articles_dump_file(wiki_dir)
    # examine_wiki_db(wiki_dir)
    # temp()
    make_page_article_similarity_distribution_plots(int(sys.argv[1]))
    end_time = time.time()
    time_elapsed = np.round((end_time - start_time) / 60, 2)
    print("Time elapsed = " + str(time_elapsed) + " minutes.")
"""
1, 106, 43475, 18,519,145
"""