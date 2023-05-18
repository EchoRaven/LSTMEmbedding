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


class WordEmbedding(nn.Module):
    def __init__(self,
                 hidden_size,
                 embed_dim,
                 num_layer,
                 word_size,
                 padding_idx=0,
                 droprate = 0.5
                 ):
        super(WordEmbedding, self).__init__()
        # 对字符做embedding
        self.embedding = nn.Embedding(num_embeddings=word_size,
                                      embedding_dim=hidden_size,
                                      padding_idx=padding_idx)

        self.encoder = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=num_layer,
                               batch_first=True,
                               dropout=droprate)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, ipt):
        ipt = self.embedding(ipt)
        output, _ = self.encoder(ipt)
        # 通过全连接层
        output = self.activate(self.fc(output))
        return output[:, -1, :]


class WordDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_size,
                 embed_dim,
                 num_layer,
                 droprate):
        super(WordDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size=in_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layer,
                               batch_first=True,
                               dropout=droprate)
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, ipt, seqlen):
        output, _ = self.decoder(ipt.unsqueeze(1).repeat(1, seqlen, 1))
        output = self.activate(self.fc(output))
        return output


class WordSeq2Seq(nn.Module):
    def __init__(self,
                 embedding_hidden_size,
                 embedding_embed_dim,
                 embedding_num_layer,
                 word_size,
                 padding_idx,
                 embedding_droprate,

                 decoder_in_dim,
                 decoder_hidden_size,
                 decoder_embed_dim,
                 decoder_num_layer,
                 decoder_droprate
                 ):
        super(WordSeq2Seq, self).__init__()
        self.encoder = WordEmbedding(embedding_hidden_size,
                                     embedding_embed_dim,
                                     embedding_num_layer,
                                     word_size,
                                     padding_idx,
                                     embedding_droprate)

        self.decoder = WordDecoder(decoder_in_dim,
                                   decoder_hidden_size,
                                   decoder_embed_dim,
                                   decoder_num_layer,
                                   decoder_droprate)

    def forward(self, ipt, seqlen):
        embedding = self.encoder(ipt)
        output = self.decoder(embedding, seqlen)
        return output
