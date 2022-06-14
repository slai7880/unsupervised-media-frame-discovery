import sys, os, pathlib, json, requests, pickle, time, h5py
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import wikipediaapi
from tqdm import tqdm, trange
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from pprint import pprint
import h5py
import networkx
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.sep, "scratch", "lais823", "HuggingfaceTransformersCache")
pathlib.Path(os.path.join(os.sep, "scratch", "lais823", "HuggingfaceTransformersCache")).mkdir(parents = True, exist_ok = True)
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

MEDIAWIKI_URL = "https://en.wikipedia.org/w/api.php"

PROJECT_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(PROJECT_DIR, "..", "GenericFraming", "data")
WIKI_DATA_DIR = os.path.join(DATA_DIR, "Wiki")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
GRAPH_DIR = os.path.join(PROJECT_DIR, "graphs")
DUMPS_TIME = "June-2021"
GENSIM_DIR = os.path.join(PROJECT_DIR, "Gensim")

NEWS_FILEPATH = os.path.join(DATA_DIR, "News", "GunViolence", "final_gv_fulltext_url.csv")
DOC2VEC_MODEL_TYPE = "PV-DBOW"
GV_FRAMES = ["Gun Rights", "Gun Control", "Politics", "Mental Health", "Public Safety", "Ethnicity", "Public Opinion", "Society/Culture", "Economic Consequence"]

STANZA_DIR = os.path.join(os.sep, "scratch", "lais823", "stanza_resources")

BERT_MODEL_TYPE = "bert-base-uncased"

def get_subcategories(category, max_attempts = 10):
    attempts, results = 0, None
    while attempts < max_attempts and results is None:
        try:
            params = {
                "action": "query",
                "cmtitle": category,
                "cmtype": "subcat",
                "cmlimit": "max",
                "list": "categorymembers",
                "format": "json"
            }
            session = requests.Session()
            response = session.get(url = MEDIAWIKI_URL, params = params)
            data = response.json()
            results = data["query"]["categorymembers"]
            print(len(results))
        except:
            attempts += 1
    return results

def check_parent_category(wiki, category, parent, maxAttempts = 10):
    attempts, result = 0, None
    while attempts < maxAttempts and result is None:
        try:
            page = wiki.page(category)
            result = parent in page.categories
        except:
            attempts += 1
    return result

def is_hidden_category(category, max_attempts = 10):
    attempts, result = 0, None
    while attempts < max_attempts and result is None:
        try:
            session = requests.Session()
            params = {
                "action": "query",
                "titles": category,
                "prop": "pageprops",
                "format": "json"
            }
            response = session.get(url = MEDIAWIKI_URL, params = params)
            data = response.json()
            result = "hiddencat" in data["query"]["pages"][next(iter(data["query"]["pages"]))]["pageprops"]
        except:
            attempts += 1
    return result

def is_helpful_category(wiki, category):
    return not "CatAutoTOC" in category and\
            not check_parent_category(wiki, category, "Category:Tracking categories") and\
            not check_parent_category(wiki, category, "Category:Hidden categories")
            
def cosine_similarity(args):
    v1, v2 = args
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))