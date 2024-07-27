import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import XLNetTokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df['text'].tolist(), df['label'].tolist()

def tokenize_data(texts, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def create_data_loader(inputs, masks, labels, batch_size, sampler):
    data = TensorDataset(inputs, masks, labels)
    data_sampler = sampler(data)
    data_loader = DataLoader(data, sampler=data_sampler, batch_size=batch_size)
    return data_loader
