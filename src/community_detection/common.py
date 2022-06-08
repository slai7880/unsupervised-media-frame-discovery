import sys, os, pathlib, json, requests, pickle, time, h5py
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import wikipediaapi
from tqdm import tqdm, trange
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from pprint import pprint
import h5py


MEDIAWIKI_URL = "https://en.wikipedia.org/w/api.php"

PROJECT_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
WIKI_DATA_DIR = os.path.join(DATA_DIR, "Wiki")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
GRAPH_DIR = os.path.join(PROJECT_DIR, "graphs")

NEWS_FILEPATH = os.path.join("..", "..", "..", "news", "GunViolence", "Dataset", "MasterData", "final_gv_fulltext_url.csv")

def getSubcategories(category):
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

def is_hidden_or_tracking(wiki, category):
    return not "CatAutoTOC" in category and\
            not check_parent_category(wiki, category, "Category:Tracking categories") and\
            not check_parent_category(wiki, category, "Category:Hidden categories")