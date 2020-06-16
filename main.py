# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

# from dataset import Data
from dataset import DataSampling
from model import SymptomNet

import argparse
import time

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/v2', help='data directory') 
parser.add_argument('--max_len_x1', default=64)
parser.add_argument('--max_len_x2', default=16)
parser.add_argument('--batch_size', default=256)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--epochs', default=180)
parser.add_argument('--out_channel_1', default=50)
parser.add_argument('--out_channel_2', default=2)
parser.add_argument('--pretrained_emb', default=True)
parser.add_argument('--use_attention', default=True)
parser.add_argument('--share_conv', default=False)
parser.add_argument('--use_gpu', default=True)
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_dev', default=True)
parser.add_argument('--do_test', default=True)
parser.add_argument('--checkpoints', default='./checkpoints/model/')
args = parser.parse_args()

model_path = f'{args.checkpoints}/net_{args.epochs}_0.pth'



# loaddata
# datapath = '/workspace/out/code/MIX_Symptom/data/dataset/origindataset20200427.csv'
datapath = ''

word2idP = 'data/mapping/word2id_whole.json'
word2vecP = 'data/mapping/word2vec_whole.npy'
label2idP = 'data/mapping/label2id.json'

# standardP = 'data/standard20200216.xlsx'
standardP = 'data/standard.xlsx'

batchsize = 64
print('Start!')
dataWithSampling = DataSampling(datapath, standardP, word2idP, word2vecP, batchsize)
# first_preProcess = True 
first_preProcess = False #split train/dev/test form oridataset or load immediately

savepath = 'data/dataset'

if first_preProcess: # if no train/dev/test file

    print("Parsing...")
    dataWithSampling.parser()

    print("Spliting...")
    dataWithSampling.splitDateset(0.8, 0.1, 0.1) 

    print("Saving...")
    dataWithSampling.saveSplitTrainAndDev(savepath)
    dataWithSampling.generateTest(savepath)
    #dataWithSampling.generateIdMapping(savepath)
else:

    print("Loading...")
    trainpath = 'data/dataset/train_ori.csv'
    devpath = 'data/dataset/dev_ori.csv'
    dataWithSampling.loadTrainAndDev(trainpath, devpath)


word2vec = dataWithSampling.word2vec
print(word2vec.size())

# build model
sym_net = SymptomNet(
    vocab_size=word2vec.size()[0],    # vocabulary size
    embed_size=word2vec.size()[1],    # token embedding size
    out_channel_1=args.out_channel_1,    # bi-channel conv kernel number
    out_channel_2=args.out_channel_2,    #attention kernel number
    max_len_x1=args.max_len_x1,    # symptom description length
    max_len_x2=args.max_len_x2,    # normalization symptom word length
    use_attention=args.use_attention,     # attention flag
    share_conv=args.share_conv    # share conv flag
    )
print(sym_net)

if args.pretrained_emb:
    sym_net.word_embedding.weight.data.copy_(dataWithSampling.word2vec)
else:
    nn.init.normal_(sym_net.word_embedding.weight, mean=0, std=0.01)
if args.use_attention:
    nn.init.normal_(sym_net.label_embedding.weight, mean=0, std=0.01)
#GPU
if args.use_gpu:
    sym_net.cuda()
#loss and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(sym_net.parameters(), lr=args.lr)

def evaluate(data_iter, net):
    loss_eva, n, eva_start = 0, 0, time.time()
    with torch.no_grad():
        for x1_, x2_, l_, y_ in data_iter:
            if args.use_gpu:
                x1_ = x1_.cuda()
                x2_ = x2_.cuda()
                l_ = l_.cuda()
                y_ = y_.cuda()
            y_hat_ = net(x1_, x2_, l_)

            loss_eva += loss(y_hat_, y_.float().view(-1, 1)).cpu().item()
            n += 1

    return loss_eva / n, time.time() - eva_start

if args.do_train:

    for epoch in range(args.epochs):
        loss_sum, batch_count, start = 0, 0, time.time()
        # nagetive sampling
        
        dataWithSampling.sampleForNeg(mode='train') # RS

        # dataWithSampling.sampleForNeg_R1(mode='train')   # DSS

        # torch.dataset
        train_data = dataWithSampling.generateDataset(label2idP, args.max_len_x1, args.max_len_x2, mode = 'train')
            
        for X1, X2, label, y in train_data:
            if args.use_gpu:
                X1 = X1.cuda()
                X2 = X2.cuda()
                label = label.cuda()
                y = y.cuda()
            y_hat = sym_net(X1, X2, label)
            l = loss(y_hat, y.float().view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l.cpu().item()

            batch_count += 1

        train_loss = loss_sum / batch_count
        use_time = time.time() - start
        print(f'Train Epoch {epoch}, Loss {train_loss}, Time {use_time} sec')

        if args.do_dev and (epoch + 1) % 2 == 0: #dev loss
            
            # nagetive sampling
            
            dataWithSampling.sampleForNeg(mode='dev') #RS

            # dataWithSampling.sampleForNeg_R1(mode='dev') #DSS
            
            # torch.dataset
            dev_data = dataWithSampling.generateDataset(label2idP, args.max_len_x1, args.max_len_x2, mode = 'dev')
            dev_loss, dev_time = evaluate(dev_data, sym_net)
            print(f'Dev Epoch {epoch}, Loss {dev_loss}, Time {dev_time} sec')
        
        if (epoch + 1) % 10 == 0: # save checkpoint
            mid_model_path = f'{args.checkpoints}/net_{epoch + 1}_0.pth'
            torch.save(sym_net.state_dict(), mid_model_path)
            print(f'model saved to {mid_model_path}.')
        
    torch.save(sym_net.state_dict(), model_path)
    print(f'model saved to {model_path}.')
