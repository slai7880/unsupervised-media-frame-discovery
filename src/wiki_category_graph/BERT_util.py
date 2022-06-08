from common import *
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.sep, "scratch", "lais823", "HuggingfaceTransformersCache")

def get_BERT_embeddings(texts, show_progress = True):
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
    if show_progress:
        for step, batch in enumerate(tqdm(dataloader)):
            torch.cuda.empty_cache()
            with torch.no_grad():
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
            embeddings.append(outputs[0][:,0,:].detach().cpu().numpy().squeeze())
    else:
        for step, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            with torch.no_grad():
                input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
                outputs = model(input_ids, token_type_ids = None, attention_mask = attention_mask)
            embeddings.append(outputs[0][:,0,:].detach().cpu().numpy().squeeze())
    return embeddings