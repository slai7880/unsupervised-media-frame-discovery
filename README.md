# Unsupervised Media Frame Discovery

This repository stores code and results presented in the paper An Unsupervised Approach to Discover Media Frames. The paper is accepted in the PoliticalNLP workshop from LREC 2022.

***

The implementation requires a few key python packages: [wikipedia-api](https://pypi.org/project/Wikipedia-API/), [Hugggingface transformers](https://huggingface.co/docs/transformers/index), and [stanza](https://stanfordnlp.github.io/stanza/).

It also requires a local Wikipedia dump file which can be found [here](https://dumps.wikimedia.org/backup-index.html). You will need a file that contains Wikipedia article pages and its name should contains "pages-articles.xml.bz2" as suffix. The dump name may change by the database administration.

The excution involves several steps.

First, go to src/wikipedia-category-graph/.

0. Open common.py file and change the directory paths accordingly. Especially the paths involving the key packages listed above.

1. The first step is to build a Doc2Vec model based on the Wikipedia articles and store the article embeddings. These can be done in one step with the Gensim Doc2Vec module.

	`python vectorize.py -b`
    
2. Next, create embeddings for news articles.

    `python vectorize.py -n`
