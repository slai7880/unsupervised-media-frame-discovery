# Unsupervised Media Frame Discovery

This repository stores code and results presented in the paper An Unsupervised Approach to Discover Media Frames. The paper is accepted in the PoliticalNLP workshop from LREC 2022.

***

The implementation requires a few key python packages: [wikipedia-api](https://pypi.org/project/Wikipedia-API/), [Hugggingface transformers](https://huggingface.co/docs/transformers/index), and [stanza](https://stanfordnlp.github.io/stanza/).

It also requires a local Wikipedia dump file which can be found [here](https://dumps.wikimedia.org/backup-index.html). You will need a file that contains Wikipedia article pages and its name should contains "pages-articles.xml.bz2" as suffix. The dump name may change by the database administration.

The excution involves several steps.

First, go to src/wikipedia_category_graph/.

0. Open common.py file and change the directory paths accordingly. Especially the paths involving the key packages listed above.

1. Build a Doc2Vec model based on the Wikipedia articles and store the article embeddings. These can be done in one step with the Gensim Doc2Vec module. **Note that this step is very time consuming.**

	`python vectorize.py -b`
    
2. Create embeddings for news articles.

    `python vectorize.py -n`
	
3. Find top ten most similar Wikipedia articles for each news article.

	`python wiki_category_graph.py -f`
	
4. Explore and fetch categories recursively. **Note that this step is very time consuming.**

	`python wiki_category_graph.py -e`
	
5. Clean and merge the categories.

	`python wiki_category_graph.py -c`
	
6. Create a graph for communiti detection.

	`python create_graphs.py`
	
Now, go to src/community_detection/.

7. Perform community detection. The results can be found in /tables/community_detection_results/.

	`python main.py`
	
Return to src/wikipedia_category_graph/.

8. Perform a cross validation of the news headline frame classification using BERT. This is not part of the pipeline, but it tells how good BERT is on the task.

	`python predict_sentence_frames.py -cv`
	
9. Train a BERT model on the news headlines. **Note that this step is very time consuming.**

	`python predict_sentence_frames.py -train`
	
10. Predict the sentence frames. The results can be found in /tables/sentence_frames/.

	`python predict_sentence_frames.py -predict`
	
11. Compare the community frames with the sentence ones. The results can be found in /tables/evaluation/.

	`python evaluate.py`
	
12. Additionally, the code used to plot the figures presented is included in this repository as well. The results can be found in /figures/.

	`python create_misc_plots.py`
	
***

## Known Issues

1. The implementation fetches the categories from the online Wikipedia database. Therefore, the resulting categories can be different if the process is done at a different time. Furthermore, we've noticed that some categories cannot be fetched even though they can be found by browsing on Wikipedia website. To address this, we are trying to change to read an offline database. However, this means that one must build an entire Wikipedia database on their local machine, which is extensively time- and space-consuming. A walkaround is to start from step 5 or 6, since the intermediate results prior to step 5 are included in this repository.

2. There are two parts in this implementation involving randomization that is not seeded. The first part is the VEC alrogithm and the other part is the BERT model. More precisely, we dicover that even though seeds are applied in both parts, one needs to dive deeper into the algorithm and the process to control the randomization. Fortunately, the overall results do not seem very different in each run.