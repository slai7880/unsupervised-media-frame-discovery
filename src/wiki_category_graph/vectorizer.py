from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
import gensim
import wikipediaapi
from multiprocessing import Pool
from time import sleep

from common import *
from WikiCategoryGraph import *

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c for c in content], [title])

def build_doc2vec_model(input_dir, output_dir, processes):
    print("Reading Wikipedia dump file.")
    time_start = time.time()
    wiki = WikiCorpus(os.path.join(input_dir, "enwiki-latest-pages-articles.xml.bz2"))
    time_end = time.time()
    time_elapsed = np.round((time_end - time_start) / 60, 2)
    print("Complete! Time elapsed: " + str(time_elapsed) + " minutes.\n")
    
    documents = TaggedWikiDocument(wiki)
    epochs = 10
    
    # PV-DBOW
    # 3.8.3
    '''
    vectorizer = Doc2Vec(dm = 0, dbow_words = 1, size = 200, window = 8, min_count = minCount, workers = processes)
    
    print("Building vocabulary with " + str(processes) + " processes.")
    time_start = time.time()
    vectorizer.build_vocab(documents)
    time_end = time.time()
    time_elapsed = np.round((time_end - time_start) / 60, 2)
    print("Complete! Time elapsed: " + str(time_elapsed) + " minutes.\n")
    
    print("Training. (Epochs = " + str(epochs) + ")")
    time_start = time.time()
    vectorizer.train(documents = documents, total_examples = vectorizer.corpus_count, epochs = epochs)
    time_end = time.time()
    time_elapsed = np.round((time_end - time_start) / 60, 2)
    print("Complete! Time elapsed: " + str(time_elapsed) + " minutes.\n")
    '''
    
    # 4.1.2
    vectorizer = Doc2Vec(documents = documents, dm = 0, dbow_words = 1, vector_size = 200, window = 8, min_count = 19, workers = processes)
    
    print("Building vocabulary with " + str(processes) + " processes.")
    time_start = time.time()
    vectorizer.build_vocab(corpus_iterable = documents)
    time_end = time.time()
    time_elapsed = np.round((time_end - time_start) / 60, 2)
    print("Complete! Time elapsed: " + str(time_elapsed) + " minutes.\n")
    
    print("Training. (Epochs = " + str(epochs) + ")")
    time_start = time.time()
    vectorizer.train(corpus_iterable = documents, total_examples = vectorizer.corpus_count, epochs = epochs)
    time_end = time.time()
    time_elapsed = np.round((time_end - time_start) / 60, 2)
    print("Complete! Time elapsed: " + str(time_elapsed) + " minutes.\n")
    
    vectorizer.save(os.path.join(output_dir, "EN_Wiki_Pages_Articles.model"), pickle_protocol = pickle.DEFAULT_PROTOCOL)
    
    # PV-DM w/average
    # vectorizer = Doc2Vec(dm = 1, dm_mean = 1, size = 200, window = 8, min_count = minCount, workers = processes)
    
    return vectorizer
        

def vectorize_GV_news(vectorizer, output_dir):
    df = pandas.read_csv(NEWS_FILEPATH)
    IDs = df["ID"].values
    texts = df["whole_news"].values.tolist()
    tokens = [list(tokenize(text, deacc = True)) for text in texts]
    news_vectors = []
    for i in trange(len(tokens)):
        vector = vectorizer.infer_vector(tokens[i])
        news_vectors.append(vector)
    news_vectors = np.array(news_vectors)
    with h5py.File(os.path.join(output_dir, "NewsVectors.h5"), "w") as h5f:
        h5f.create_dataset("NewsVectors", data = news_vectors)

        

if __name__ == "__main__":
    wiki_dir = os.path.join(WIKI_DATA_DIR, "Dumps", DUMPS_TIME)
    model_dir = os.path.join(GENSIM_DIR, "GensimModels", "Doc2Vec " + str(gensim.__version__), "Wiki-" + DUMPS_TIME, DOC2VEC_MODEL_TYPE)
    pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
    if len(sys.argv) > 1:
        if "-b" in sys.argv:
            vectorizer = build_doc2vec_model(wiki_dir, model_dir, 1)
        if "-n" in sys.argv:
            vectorizer = Doc2Vec.load(os.path.join(model_dir, "EN_Wiki_Pages_Articles.model"))
            output_dir = os.path.join(GENSIM_DIR, "GensimEmbeddings", DUMPS_TIME, "GunViolence", DOC2VEC_MODEL_TYPE)
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            vectorize_GV_news(vectorizer, output_dir)