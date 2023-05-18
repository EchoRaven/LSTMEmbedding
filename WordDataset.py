from abc import ABC
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Tokenizer
import json
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np


def getlist(file):
    wordjson = json.load(open(file, encoding="utf-8"))
    wordlist = []
    alphalist = []
    for i in wordjson:
        wordlist.append(i)
    for w in wordlist:
        for a in w:
            if a not in alphalist:
                alphalist.append(a)
    # 按照ascii排序
    alphalist.sort()
    return wordlist, alphalist


# 构建数据集（其实就是GPT2的词表）
class WordDataset(Dataset, ABC):
    def __init__(self, file, tokenizer, maxlen=None):
        wordlist, alphalist = getlist(file)
        self.dataset = []
        if maxlen is None:
            maxlen = 0
            for i in wordlist:
                maxlen = max(maxlen, len(i))
        self.maxlen = maxlen
        for i in wordlist:
            token = tokenizer.encode(i)
            token = torch.cat([token, torch.zeros(maxlen - len(token))], dim=0).long()
            self.dataset.append(token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# （构建分词器，其实就是为字母编码）
class WordTokenizer:
    def __init__(self, file):
        wordlist, alphalist = getlist(file)
        self.alphalist = ["[PAD]", "[MSK]", "[EOS]", "[CLS]", "[SEP]", "[UNK]"]
        self.alphalist.extend(alphalist)
        self.wordsize = len(self.alphalist)

    def encode(self, ipt):
        res = torch.LongTensor(len(ipt))
        for i in range(len(ipt)):
            res[i] = self.alphalist.index(ipt[i])
        return res

    def decode(self, ipt):
        res = ""
        for i in ipt:
            res += self.alphalist[i]
        return res

    def __len__(self):
        return len(self.alphalist)
