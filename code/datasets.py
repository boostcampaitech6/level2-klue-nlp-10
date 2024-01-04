import pickle as pickle
import os
import pandas as pd
import torch
from typing import List


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def tokenized_dataset(tokenizer, prompt:List , sentence:List, max_length:int=256):
    """prompt와 sentence 입력시 tokenizer에 따라 sentence를 tokenizing 하는 메서드."""

    tokenized_sentences = tokenizer(prompt,
                                    sentence,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=max_length,
                                    add_special_tokens=True,
                                    )
    return tokenized_sentences
