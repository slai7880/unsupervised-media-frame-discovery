from common import *
import torch, gc
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import StratifiedKFold
import re
from multiprocessing import Pool
import nltk.data

def get_headline_data(use_master_file = True, include_no_frame = True):
    if use_master_file:
        df_news = pandas.read_excel(os.path.join(DATA_DIR, "News", "GunViolence", "Gun violence_master file.xlsx"))
    else:
        df_news = pandas.read_csv(os.path.join(DATA_DIR, "News", "GunViolence", "final_gv_fulltext_url.csv"))
    headlines, headline_frames = [], []
    for i in range(df_news.shape[0]):
        if df_news["Q3 Theme1"][i] == 99:
            if include_no_frame:
                headlines.append(df_news["news_title"][i].lower())
                headline_frames.append(0)
        else:
            headlines.append(df_news["news_title"][i].lower())
            headline_frames.append(df_news["Q3 Theme1"][i])
    return np.array(headlines), np.array(headline_frames)

def select(source, indices):
    return [source[i] for i in indices]

def get_sentence_data(min_sentence_length = 20):
    df_news = pandas.read_csv(os.path.join(DATA_DIR, "News", "GunViolence", "final_gv_fulltext_url.csv"))
    sentences = []
    news_IDs = []
    headlines = []
    headline_frames = []
    URLs = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i in range(df_news.shape[0]):
        temp = tokenizer.tokenize(df_news["whole_news"][i])
        sentences += temp
        news_IDs += [df_news["ID"][i]] * len(temp)
        headlines += [df_news["news_title"][i]] * len(temp)
        f = df_news["Q3 Theme1"][i]
        if f == 99:
            f = 0
        headline_frames += [f] * len(temp)
        URLs += [df_news["URL"][i]] * len(temp)
    tuples = [(i, len(s), s) for (i, s) in enumerate(sentences)]
    tuples.sort(key = lambda x : x[1])
    start = 0
    while start < len(tuples) and tuples[start][1] < min_sentence_length: # this is an artibrary number
        start += 1
    indices = [t[0] for t in tuples[start:]]
    indices.sort()
    sentences = select(sentences, indices)
    sentences = [s.lower() for s in sentences]
    news_IDs = select(news_IDs, indices)
    headlines = select(headlines, indices)
    headline_frames = select(headline_frames, indices)
    URLs = select(URLs, indices)
    return np.array(sentences), np.array(news_IDs), np.array(headlines), np.array(headline_frames), np.array(URLs)
    
def cross_validate(headlines, headline_frames, model_type, epochs, batch_size, learning_rate, output_dir = None):
    print("Cross validating.")
    print("Hyper-parameters:")
    print("epochs = " + str(epochs) + "  batch_size = " + str(batch_size) + "  learning_rate = " + str(learning_rate))
    num_labels = len(set(headline_frames))
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)
    
    device = torch.device("cuda")
    torch.manual_seed(4)
    torch.cuda.manual_seed_all(4)
    tokenizer = BertTokenizer.from_pretrained(model_type)
    YTrue_allsplits, YPredict_allsplits = [], []
    counter = 0
    for train, test in skf.split(headlines, headline_frames):
        print("Running on split " + str(counter) + ".")
        
        print("Training.")
        encoding_train = tokenizer.batch_encode_plus(headlines[train], add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
        Y_train_tensor = torch.tensor(headline_frames[train])
        dataset_train = TensorDataset(encoding_train["input_ids"], encoding_train["attention_mask"], Y_train_tensor)
        dataloader_train = DataLoader(dataset_train, sampler = RandomSampler(dataset_train), batch_size = batch_size)
        
        model = BertForSequenceClassification.from_pretrained(model_type, num_labels = num_labels, output_attentions = False, output_hidden_states = False)
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr = learning_rate)
        total_steps = len(dataloader_train) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        
        for epoch in trange(epochs):
            model.train()
            for step, batch in enumerate(dataloader_train):
                torch.cuda.empty_cache()
                input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                model.zero_grad()
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask, labels = labels)
                loss = outputs[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        
        print("Validating.")
        encoding_test = tokenizer.batch_encode_plus(headlines[test], add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
        Y_test_tensor = torch.tensor(headline_frames[test])
        dataset_test = TensorDataset(encoding_test["input_ids"], encoding_test["attention_mask"], Y_test_tensor)
        dataloader_test = DataLoader(dataset_test, sampler = SequentialSampler(dataset_test), batch_size = 1)
        YPredict = []
        for step, batch in enumerate(dataloader_test):
            torch.cuda.empty_cache()
            with torch.no_grad():
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
            YPredict.append(np.argmax(outputs[0].detach().cpu().numpy()[0]))
        
        YTrue_allsplits.append(headline_frames[test])
        YPredict_allsplits.append(YPredict)
        
        counter += 1
    
    if output_dir:
        pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
        with open(os.path.join(output_dir, "YTrueAllSplits.pkl"), "wb") as f:
            pickle.dump(YTrue_allsplits, f, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(output_dir, "YPredictAllSplits.pkl"), "wb") as f:
            pickle.dump(YPredict_allsplits, f, pickle.DEFAULT_PROTOCOL)
    accuracy, f1 = accuracy_score(np.concatenate(YTrue_allsplits), np.concatenate(YPredict_allsplits)), f1_score(np.concatenate(YTrue_allsplits), np.concatenate(YPredict_allsplits), average = "micro")
    print("Accuracy = " + str(accuracy))
    print("F1 = " + str(f1))
    print()
    
    '''
    Accuracy = 0.8120401337792642
    F1 = 0.8120401337792642
    '''
    return accuracy, f1

# Only gun violence data is available
def train_BERT_model(headlines, headline_frames, model_type, epochs, batch_size, learning_rate, model_dir = None):
    print("Training on all headlines.")
    num_labels = len(set(headline_frames))
    
    device = torch.device("cuda")
    torch.manual_seed(4)
    torch.cuda.manual_seed_all(4)
    tokenizer = BertTokenizer.from_pretrained(model_type)
    encoding = tokenizer.batch_encode_plus(headlines, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    
    
    Y_tensor = torch.tensor(headline_frames)
    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"], Y_tensor)
    dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size)
    
    model = BertForSequenceClassification.from_pretrained(model_type, num_labels = num_labels, output_attentions = False, output_hidden_states = False)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    for epoch in range(epochs):
        print("Epoch " + str(epoch) + " / " + str(epochs - 1))
        print("Training....")
        start_time = time.time()
        model.train()
        for step, batch in enumerate(tqdm(dataloader)):
            torch.cuda.empty_cache()
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.zero_grad()
            outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask, labels = labels)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        if model_dir:
            pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))


def predict_sentence_frames_with_BERT(sentences, news_IDs, headlines, headline_frames, URLs,
                                        model_type, model_dir, output_dir):
    # device = torch.device("cpu")
    device = torch.device("cuda")
    
    model = BertForSequenceClassification.from_pretrained(model_type, num_labels = 10, output_attentions = False, output_hidden_states = False)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(model_type)
    encoding = tokenizer.batch_encode_plus(sentences, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    
    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
    dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)
    
    YPredict = []
    for step, batch in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
        YPredict.append(np.argmax(outputs[0].detach().cpu().numpy()[0]))
    
    df_out = pandas.DataFrame({"Headline" : headlines, "ID" : news_IDs, "Headline Frame" : headline_frames, "URL" : URLs, "Sentence" : sentences, "Frame" : YPredict})
    df_out.to_csv(os.path.join(output_dir, "SentenceFrames.csv"), index = False)
    
if __name__ == "__main__":
    model_type = BERT_MODEL_TYPE
    headlines, headline_frames = get_headline_data()
    cache_dir = os.path.join(CACHE_DIR, "predict_sentence_frames", "GunViolence")
    
    if len(sys.argv) > 1:
        if "-cv" in sys.argv:
            headline_frames_train = headline_frames
            output_dir = os.path.join(cache_dir, "StratifiedKFoldCVOutputs")
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            epochs_list = [6, 7, 8, 9, 10, 11, 12, 13, 14]
            learning_rate_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]
            epochs_list, learning_rate_list = epochs_list[4 : 7], learning_rate_list[:2] # for narrowing range purpose
            batch_size = 4
            
            M = {"Set" : [], "Epochs" : [], "Batch Size" : [], "Learning Rate" : [], "Accuracy" : [], "F1" : []}
            
            count = 0
            for epochs in epochs_list:
                for learning_rate in learning_rate_list:
                    accuracy, f1 = cross_validate(headlines, headline_frames, model_type, epochs, batch_size, learning_rate, os.path.join(output_dir, str(count)))
                    # accuracy, f1 = 0, 0
                    M["Set"].append(count)
                    M["Epochs"].append(epochs)
                    M["Batch Size"].append(batch_size)
                    M["Learning Rate"].append(learning_rate)
                    M["Accuracy"].append(accuracy)
                    M["F1"].append(f1)
                    count += 1
            df = pandas.DataFrame(M)
            df.to_csv(os.path.join(output_dir, "CVSummary.csv"), index = False)
        
        if "-train" in sys.argv:
            # optimal hyper-parameter valuee
            epochs = 12
            batch_size = 4
            learning_rate = 2e-5
            
            model_dir = os.path.join(cache_dir, "BERT_model_headlines")
            train_BERT_model(headlines, headline_frames, model_type, epochs, batch_size, learning_rate, model_dir)
        
        if "-predict" in sys.argv:
            sentences, news_IDs, headlines, headline_frames, URLs = get_sentence_data()
            model_dir = os.path.join(cache_dir, "BERT_model_headlines")
            output_dir = os.path.join(TABLE_DIR, "sentence_frames", "GunViolence")
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            predict_sentence_frames_with_BERT(sentences, news_IDs, headlines, headline_frames, URLs, model_type, model_dir, output_dir)
    