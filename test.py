# -*-coding:utf-8 -*-
import torch
import json
import numpy as np
from convert2vec import Converter

from model import SymptomNet

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

use_gpu = True
batch_number = 5

model_path = sys.argv[1]
converter = Converter()
print(model_path)

# load model
sym_net = SymptomNet(vocab_size=15233, embed_size=300, out_channel_1=50, out_channel_2=2,
    max_len_x1=64, max_len_x2=16, use_attention=True, share_conv=False)
sym_net.load_state_dict(torch.load(model_path))
if use_gpu:
    sym_net.cuda()

# load data
predict = []
testfilePath = 'data/dataset/predict.txt'
with open(testfilePath, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        predict.append(line.strip().split('\t'))
bz2vec = np.load('data/mapping/HSI-Sym-id/bz2vec.npy')
bz2id = json.load(open("data/mapping/HSI-Sym-id/bz2id.json"))
label2id = json.load(open("data/mapping/HSI-Sym-id/label2id.json"))
label2bzid = json.load(open("data/mapping/HSI-Sym-id/label_bz2id.json"))

print(bz2vec.shape)

# test
bz_n = bz2vec.shape[0]
batch_size = bz_n // batch_number
hits, hits_f, hits_l = [0] * bz_n, [0] * bz_n, [0] * bz_n
meanrank, meanrank_f, meanrank_l = [], [], []
for sample in predict:
    sample = [sample[0].replace(" ", ""), sample[1], sample[2]]
    label = torch.from_numpy(np.array([[label2id[sample[1]]]] * bz_n))
    x1 = torch.from_numpy(np.array([converter.convert(sample[0], maxlen=64)] * bz_n))
    x2 = torch.from_numpy(bz2vec)
    scores = []
    for batch in range(batch_number + 1):
        x1_ = x1[batch * batch_size: (batch + 1) * batch_size, :]
        x2_ = x2[batch * batch_size: (batch + 1) * batch_size, :]
        label_ = label[batch * batch_size: (batch + 1) * batch_size, :]
        if use_gpu:
            x1_ = x1_.cuda()
            x2_ = x2_.cuda()
            label_ = label_.cuda()
        results = sym_net(x1_, x2_, label_)
        # print(results)
        start_index = batch * batch_size
        scores += [(index + start_index, r[0]) for index, r in enumerate(results.cpu().detach().numpy().tolist())]
    scores.sort(key=lambda x: x[1], reverse=True)
    bz_id = bz2id[sample[2]]
    filter_number = 0
    for i, score in enumerate(scores):
        if score[0] == bz_id:
            hits[i] += 1
            meanrank.append(i + 1)
            for j in range(i + 1):
                if scores[j][0] not in label2bzid.get(sample[1], []):
                    filter_number += 1
                if scores[j][1] == score[1]:
                    # hits_f[j] += 1
                    # meanrank_f.append(j + 1)
                    hits_l[j - filter_number] += 1
                    meanrank_l.append(j + 1 - filter_number)
                    break
            # print(f'sample - {sample} {list(bz2id.keys())[list(bz2id.values()).index(scores[0][0])]} {i + 1} {j + 1} {j + 1 - filter_number} ')
            print(f'sample - {sample} {i + 1} {j + 1} {j + 1 - filter_number}')
            break

print('normal:')
print(f'\tHits@1: {sum(hits[:1]) / len(predict)}')
print(f'\tHits@3: {sum(hits[:3]) / len(predict)}')
print(f'\tHits@10: {sum(hits[:10]) / len(predict)}')
print(f'\tMean Rank: {sum(meanrank) / len(predict)}')
print(f'\tMean Reciprocal Rank: {sum([1 / mr for mr in meanrank]) / len(predict)}')

# print('forward:')
# print(f'\tHits@1: {sum(hits_f[:1]) / len(predict)}')
# print(f'\tHits@3: {sum(hits_f[:3]) / len(predict)}')
# print(f'\tHits@10: {sum(hits_f[:10]) / len(predict)}')
# print(f'\tMean Rank: {sum(meanrank_f) / len(predict)}')
# print(f'\tMean Reciprocal Rank: {sum([1 / mr for mr in meanrank_f]) / len(predict)}')

print('label:')
print(f'\tHits@1: {sum(hits_l[:1]) / len(predict)}')
print(f'\tHits@3: {sum(hits_l[:3]) / len(predict)}')
print(f'\tHits@10: {sum(hits_l[:10]) / len(predict)}')
print(f'\tMean Rank: {sum(meanrank_l) / len(predict)}')
print(f'\tMean Reciprocal Rank: {sum([1 / mr for mr in meanrank_l]) / len(predict)}') 
