from common import *
from sklearn.metrics import adjusted_rand_score
from matplotlib.ticker import MaxNLocator

def plot_wiki_to_news_similarity_trends(cache_dir, fig_dir):
    scores = np.load(os.path.join(cache_dir, "Top100SimilarityScores.npy"))
    average_scores = np.mean(scores, axis = 0)
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(average_scores, "-o")
    ax.set_xlabel("Rank", fontsize = 16)
    ax.set_ylabel("Cosine Similarity", fontsize = 16)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "CosineSimilarityFirst100.png"))
    plt.close(fig)

def plot_adjusted_RAND_index_scores(community_excel_file, fig_dir):
    n_communities = [i for i in range(2, 21)]
    rand_indices = []
    for i in n_communities:
        df_SC = pandas.read_excel(community_excel_file, sheet_name = "SC_" + str(i) + "_comms")
        df_VEC = pandas.read_excel(community_excel_file, sheet_name = "VEC_" + str(i) + "_comms")
        labels_1, labels_2 = df_SC["Label"].values.tolist(), df_VEC["Label"].values.tolist()
        rand_indices.append(adjusted_rand_score(labels_1, labels_2))
    
    fig, axes = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
    axes[0].plot(n_communities, rand_indices, "o")
    axes[0].set_yticks(np.linspace(0, 1, 51))
    axes[1].plot(n_communities, rand_indices, "o")
    axes[1].set_yticks(np.linspace(0, 1, 11))
    
    axes[0].set_ylim(0.8, 0.84)
    axes[1].set_ylim(0, 0.3)
    axes[0].spines['bottom'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[0].xaxis.tick_top()
    axes[0].tick_params(labeltop = False)
    axes[1].xaxis.tick_bottom()
    
    d = .015
    kwargs = dict(transform = axes[0].transAxes, color = 'k', clip_on = False)
    axes[0].plot((-d, +d), (-d, +d), **kwargs)
    axes[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform = axes[1].transAxes)
    axes[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axes[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    axes[1].set_xlabel("Number of Communities", fontsize = 16)
    axes[1].set_xticks(n_communities)
    axes[1].set_xticklabels(n_communities)
    axes[0].grid(axis = "y")
    axes[1].grid(axis = "y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "AdjustedRANDIndices.png"))
    plt.close(fig)

def plot_community_sizes(community_excel_file, fig_dir):
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    max_counts = [[], []]
    for i in range(2, 21):
        df_SC = pandas.read_excel(community_excel_file, sheet_name = "SC_" + str(i) + "_comms")
        df_VEC = pandas.read_excel(community_excel_file, sheet_name = "VEC_" + str(i) + "_comms")
        labels = [df_SC["Label"].values.tolist(), df_VEC["Label"].values.tolist()]
        counts=  [[labels[0].count(j) for j in range(i)], [labels[1].count(j) for j in range(i)]]
        max_counts[0].append(max(counts[0]))
        max_counts[1].append(max(counts[1]))
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.bar(np.array([j for j in range(i)]) - 0.2, counts[0], 0.4, label = "SC")
        ax.bar(np.array([j for j in range(i)]) + 0.2, counts[1], 0.4, label = "VEC")
        ax.set_xlabel("Community Label", fontsize = 16)
        ax.set_ylabel("Count", fontsize = 16)
        ax.set_xticks([j for j in range(i)])
        ax.set_xticklabels([j for j in range(i)])
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))
        ax.legend(prop = {"size" : 16})
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, str(i) + "_comms.png"))
        plt.close(fig)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot([i for i in range(2, 21)], max_counts[0], "-o", label = "SC")
    ax.plot([i for i in range(2, 21)], max_counts[1], "-o", label = "VEC")
    ax.set_xlabel("Community Number", fontsize = 16)
    ax.set_ylabel("Greatest Community Size", fontsize = 16)
    ax.set_xticks([i for i in range(2, 21)])
    ax.set_xticklabels([i for i in range(2, 21)])
    ax.legend(prop = {"size" : 18})
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "GreatestCommunitySizes.png"))
    plt.close(fig)

def create_headline_frame_histograms(sentenece_frame_dir, fig_dir):
    all_frames = ["No Frame"] + GV_FRAMES
    df_news = pandas.read_excel(os.path.join(DATA_DIR, "News", "GunViolence", "Gun violence_master file.xlsx"))
    headline_frames = []
    for i in range(df_news.shape[0]):
        if df_news["Q3 Theme1"][i] == 99:
            headline_frames.append("No Frame")
        else:
            headline_frames.append(all_frames[df_news["Q3 Theme1"][i]])
    df_sentence_frames = pandas.read_csv(os.path.join(sentenece_frame_dir, "SentenceFrames.csv"))
    sentence_frames = []
    for i in range(df_sentence_frames.shape[0]):
        sentence_frames.append(all_frames[df_sentence_frames["Frame"][i]])
    headline_frame_counts = [headline_frames.count(i) for i in all_frames]
    sentence_frame_counts = [sentence_frames.count(i) for i in all_frames]
    fig, axes = plt.subplots(2, 1, figsize = (8, 12), sharex = True)
    axes[0].bar([i for i in range(len(all_frames))], headline_frame_counts)
    axes[0].set_ylabel("Count", fontsize = 16)
    axes[1].bar([i for i in range(len(all_frames))], sentence_frame_counts)
    axes[1].set_xlabel("Frame", fontsize = 16)
    axes[1].set_ylabel("Count", fontsize = 16)
    axes[1].set_xticks([i for i in range(len(all_frames))])
    axes[1].set_xticklabels(all_frames, rotation = 45, fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "OverallFrameHistogram.png"))
    plt.close(fig)

def create_soft_label_heatmaps(eval_dir, fig_dir):
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    mapping_excel_file = pandas.ExcelFile(os.path.join(eval_dir, "clusters_to_frames.xlsx"))
    for n in [4, 7, 11, 19]:
        for cm in ["SC", "VEC"]:
            df_mapping = pandas.read_excel(mapping_excel_file, sheet_name = cm + "_" + str(n) + "_comms")
            M = np.zeros((n, 9), dtype = "double")
            for i in range(df_mapping.shape[0]):
                for rank in range(1, 10):
                    M[i, GV_FRAMES.index(df_mapping["Frame " + str(rank)][i])] = df_mapping["Frame " + str(rank) + " score"][i]
            fig, ax = plt.subplots(figsize = (12, 10))
            sns.heatmap(M, vmin = 0.84, vmax = 0.98, xticklabels = GV_FRAMES, ax = ax)
            ax.set_ylabel("Community Label", fontsize = 16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 14)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "HeadlineFrameDistribution" + cm + str(n) + ".png"))
            plt.close(fig)

def create_error_plots(output_dir, fig_dir):
    print("Creating error plot.")
    clustering_methods = ["SC", "VEC"]
    n_communities = [i for i in range(2, 21)]
    df_error = pandas.read_csv(os.path.join(output_dir, "clusters_to_frames_errors.csv"))
    
    fig, ax = plt.subplots(figsize = (8, 6))
    for cm in clustering_methods:
        ax.plot(df_error[cm], "-o", label = cm)
    ax.grid(True)
    ax.set_xlabel("Number of Communities", fontsize = 16)
    ax.set_ylabel("Jensen Shannon Distance", fontsize = 16)
    ax.set_xticks([i for i in range(len(n_communities))])
    ax.set_xticklabels(n_communities)
    ax.legend(prop = {"size" : 16})
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "errors.png"))
    plt.close(fig)

if __name__ == "__main__":
    news = "GunViolence"
    cache_dir = os.path.join(CACHE_DIR, "wiki_category_graph", news)
    sentence_frames_dir = os.path.join(TABLE_DIR, "sentence_frames", news)
    graph_dir = os.path.join(GRAPH_DIR, news)
    community_dir = os.path.join(TABLE_DIR, "community_detection_outputs", news)
    eval_dir = os.path.join(TABLE_DIR, "evaluation")
    
    graph_file_index = 0
    with open(os.path.join(graph_dir, "graph_" + str(graph_file_index) + ".json"), "r") as f:
        graph = json.load(f)
    community_excel_file = pandas.ExcelFile(os.path.join(community_dir, "community_labels_" + str(graph_file_index) + ".xlsx"))
    output_dir = os.path.join(TABLE_DIR, "evaluation")

    fig_dir = os.path.join(FIGURE_DIR, "misc figures")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    plot_wiki_to_news_similarity_trends(cache_dir = cache_dir, fig_dir = fig_dir)
    plot_adjusted_RAND_index_scores(community_excel_file, fig_dir)
    plot_community_sizes(community_excel_file, os.path.join(fig_dir, "community size"))
    create_headline_frame_histograms(sentence_frames_dir, fig_dir)
    create_soft_label_heatmaps(eval_dir, os.path.join(fig_dir, "HeadlineFrameDistributions"))
    
    fig_dir = os.path.join(FIGURE_DIR, "evaluation", "GunViolence")
    pathlib.Path(fig_dir).mkdir(parents = True, exist_ok = True)
    create_error_plots(eval_dir, fig_dir)