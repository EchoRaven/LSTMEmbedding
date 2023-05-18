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
    def __init__(self, in_dim, hidden_size, embed_dim, num_layer, word_size):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=word_size, embedding_dim=hidden_size)
        self.encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, ipt, length):
        ipt = self.embedding(ipt)
        output, _ = self.encoder(ipt)
        b = length.size(0)
        indices = torch.arange(b)
        output = output[indices, length - 1]
        output = self.activate(self.fc(output))
        return output


class WordDecoder(nn.Module):
    def __init__(self, in_dim, hidden_size, embed_dim, num_layer):
        super(WordDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, ipt, len):
        output, _ = self.decoder(ipt.unsqueeze(1).repeat(1, len, 1))
        output = self.activate(self.fc(output))
        return output


class WordTokenizer:
    def __init__(self, alphalist):
        self.alphalist = ["[MSK]", "[EOS]", "[CLS]", "[SEP]", "[UNK]"]
        self.alphalist.extend(alphalist)

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


class WordDataset(Dataset, ABC):
    def __init__(self, wordlist, tokenizer, maxlen=None):
        self.dataset = []
        self.lengths = []
        if maxlen is None:
            maxlen = 0
            for i in wordlist:
                maxlen = max(maxlen, len(i))
        self.maxlen = maxlen
        for i in wordlist:
            self.lengths.append(len(i))
            token = tokenizer.encode(i)
            token = torch.cat([token, torch.zeros(maxlen - len(token))], dim=0).long()
            self.dataset.append(token)
        self.lengths = torch.LongTensor(np.array(self.lengths))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.lengths[idx]


class WordEncoderDecoder(nn.Module):
    def __init__(self, embedding_in_dim, embedding_hidden_size, embedding_embed_dim, embedding_num_layer, word_size,
                 decoder_in_dim, decoder_hidden_size, decoder_embed_dim, decoder_num_layer):
        super(WordEncoderDecoder, self).__init__()
        self.encoder = WordEmbedding(embedding_in_dim, embedding_hidden_size, embedding_embed_dim, embedding_num_layer, word_size)
        self.decoder = WordDecoder(decoder_in_dim, decoder_hidden_size, decoder_embed_dim, decoder_num_layer)

    def forward(self, ipt, lene, lend):
        embedding = self.encoder(ipt, lene)
        res = self.decoder(embedding, lend)
        return res


def getlist():
    file = "vocab.json"
    wordjson = json.load(open(file))
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


if __name__ == "__main__":
    wordlist, alphalist = getlist()
    wordtokenizer = WordTokenizer(alphalist)
    word_size = len(wordtokenizer)

    # 构建模型
    wordencoderdecoder = WordEncoderDecoder(1, 32, 32, 4, word_size, 32, 32, word_size, 4).cuda()
    # wordencoderdecoder = torch.load("wordEncoderDecoder.pth").cuda()
    worddataset = WordDataset(wordlist, wordtokenizer)
    batch_size = 128
    wordloader = DataLoader(worddataset, batch_size=batch_size, shuffle=True, drop_last=True)
    epoch = 4000
    maxlen = worddataset.maxlen
    optimizerWord = optim.Adam(wordencoderdecoder.parameters(), lr=1e-4, weight_decay=0.002)
    for e in range(epoch):
        for index, batch in enumerate(wordloader):
            optimizerWord.zero_grad()
            ipt, length = batch
            ipt = ipt.cuda()
            length = length.cuda()
            word = wordencoderdecoder(ipt, length, maxlen).cuda()
            targets = ipt
            mask = (targets.view(-1) != 0).float()
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            outputs = word.view(-1, word_size)
            targets = targets.view(-1)
            loss = loss_fn(outputs, targets.long())
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizerWord.step()
            if (index + 1) % 128 == 0:
                message = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(e + 1, epoch, index + 1,
                                                                             len(wordloader),
                                                                             loss.item())
                print(message)
    torch.save(wordencoderdecoder, "wordEncoderDecoder.pth")

    model = torch.load("wordEncoderDecoder.pth").cuda()
    ipt = "watermelon"
    token = wordtokenizer.encode(ipt).unsqueeze(0).cuda()
    res = torch.argmax(model(token.float(), torch.LongTensor([len(ipt)]), len(ipt)), dim=2)
    print(wordtokenizer.decode(res[0]))
