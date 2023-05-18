from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn as nn


from WordEmbedding import WordSeq2Seq
from WordDataset import WordTokenizer, WordDataset

if __name__ == "__main__":

    # 实例化分词器
    wordTokenizer = WordTokenizer("vocab.json")

    # 实例化数据集
    wordDataset = WordDataset(tokenizer=wordTokenizer,
                              file="vocab.json")
    wordsize = wordTokenizer.wordsize
    seqlen = wordDataset.maxlen

    # 实例化模型
    wordseq2seq = WordSeq2Seq(embedding_hidden_size=32,
                              embedding_embed_dim=32,
                              embedding_num_layer=4,
                              word_size=wordsize,
                              padding_idx=0,
                              embedding_droprate=0.5,

                              decoder_in_dim=32,
                              decoder_hidden_size=32,
                              decoder_embed_dim=wordsize,
                              decoder_num_layer=4,
                              decoder_droprate=0.5).cuda()


    # 设置训练参数
    batch_size = 128
    print_freq = 128
    epoch = 100
    lr = 1e-4
    weight_decay = 0.002
    lossFunc = nn.CrossEntropyLoss(reduction='none')

    # 构建迭代器
    wordloader = DataLoader(wordDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 优化器
    optimizer = optim.Adam(wordseq2seq.parameters(), lr=lr, weight_decay=weight_decay)
    # 训练模型
    for e in range(epoch):
        for index, batch in enumerate(wordloader):
            target = batch.cuda()
            output = wordseq2seq(target, seqlen)
            # 获取mask
            mask = (target.view(-1) != 0).float().cuda()
            output = output.view(-1, wordsize)
            target = target.view(-1)
            loss = lossFunc(output, target.long())
            loss = (loss * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (index + 1) % print_freq == 0:
                message = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(e + 1, epoch, index + 1,
                                                                             len(wordloader),
                                                                             loss.item())
                print(message)

    # 保存模型
    torch.save(wordseq2seq, "wordseq2seq.pth")
