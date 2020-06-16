# -*-coding:utf-8 -*-

import torch
import numpy as np
import random
from torch.utils.data import Dataset as TorchDataset

import math
import csv
import json

from utils.ExcelTool import ExcelReader

# add for chinese utf-8
# import sys
# import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

class CSVReader(object):
    def __init__(self, filename):
        self.__reader = csv.reader(open(filename, 'r', encoding='utf-8'))

    def getData(self):
        data = []
        for row in self.__reader:
            data.append(row)
        return data


class CSVWriter(object):
    def __init__(self, filename):
        self.__writer = csv.writer(open(filename, 'a', encoding='utf-8', newline=''), dialect='excel')

    def writeData(self, results):
        for row in results:
            if len(row) != 0:
                self.__writer.writerow(row)

class DataSampling():
    TRAIN_MODE = 'train'
    DEV_MODE = 'dev'
    TEST_MODE = 'test'

    def __init__(self, origindatapath, standardpath, word2idpath, word2vecpath, batch_size):
        super().__init__()

        self.TRAIN_MODE = 'train'
        self.DEV_MODE = 'dev'
        self.TEST_MODE = 'test'
        
        self.origindatapath = origindatapath
        # the whole data
        self.Positive = list()
        self.Negtive = list()
        # train
        self.Positive_train = list()
        self.Negtive_train = list()
        self.SamplingNegtive_train = list()
        # dev
        self.Positive_dev = list()
        self.Negtive_dev = list()
        self.SamplingNegtive_dev = list()
        # test
        self.Positive_test = list()
        self.Negtive_test = list()
        self.SamplingNegtive_test = list()

        self.SamplingNegtive = list()
        
        self.standardPath = standardpath
        self.zz2label = dict() # use to generate Train/Dev/Test dataset

        # get standard word to id mapping
        data = ExcelReader(self.standardpath).getSheetData('Sheet1')[1:]
        #symptom2 Hierachal semantic infomation(HSI)label mapping
        for line in data:
            if line[3] == 'None' or line[1] == 'None':
                continue
            self.zz2label[line[3]] = line[1]

        # load word2vec
        self.word2idpath = word2idpath
        self.word2id = json.load(open(word2idpath, encoding='utf-8'))
        self.word2vecpath = word2vecpath
        self.word2vec = torch.from_numpy(np.load(word2vecpath))
        # for saving torch.Dataset
        self.batch_size = batch_size
        self.dataset_gen = None
        self.iter_dateset_gen =None

        # self._parser(self)
    
    # def get_zz2label(self):
    #     if self.standardpath != '':
    #         # get standard word to id mapping
    #         data = ExcelReader(self.standardpath).getSheetData('Sheet1')[1:]
    #         #symptom2 Hierachal semantic infomation(HSI)label mapping
    #         for line in data:
    #             if line[3] == 'None' or line[1] == 'None':
    #                 continue
    #             self.zz2label[line[3]] = line[1]
    #     else:
    #         print('No standardpath!')
        
    
    # parser the whole dataset
    def parser(self):
        # clear first
        self.Positive.clear()
        self.Negtive.clear()
        # parsing... positive index = nagtive index
        data = CSVReader(self.origindatapath).getData()[1:] #table head skip(descroption of col)
        neglist = list()
        for index in range(len(data)-1, -1, -1): 
            score = float(data[index][2])
            if math.isclose(score,1.0):
                self.Positive.append([data[index][0],data[index][1],score])                
                self.Negtive.append(sorted(neglist, key = lambda x:x[2],reverse=True))
                neglist.clear()
            else:
                neglist.append([data[index][0],data[index][1],score])
        if len(self.Negtive) == len(self.Positive):
            print("Good parsing.")
        else:
            print("Error on mapping.")

    #split ori dataset according ration (train+dev+test=1)
    def splitDateset(self, trainRatio, devRatio, testRatio):
        # clean
        self.Positive_train.clear()
        self.Negtive_train.clear()
        
        self.Positive_dev.clear()
        self.Negtive_dev.clear()
        
        self.Positive_test.clear()
        self.Negtive_test.clear()

        # index get
        indexlist = [i for i in range(0, len(self.Positive))]
        random.shuffle(indexlist)
        trainIndex = indexlist[:int(trainRatio*len(self.Positive))]
        devIndex = indexlist[int(trainRatio*len(self.Positive)):int((trainRatio+devRatio)*len(self.Positive))]
        testIndex = indexlist[int((trainRatio+devRatio)*len(self.Positive)):]


        # gen data
        # train
        for indextrain in trainIndex:
            if len(self.Negtive[indextrain]) == 0: 
                self.Positive_test.append(self.Positive[indextrain])
                self.Negtive_test.append(self.Negtive[indextrain])
            else:
                self.Positive_train.append(self.Positive[indextrain])
                self.Negtive_train.append(self.Negtive[indextrain])
        
        # dev
        for indexdev in devIndex:
            if len(self.Negtive[indexdev]) == 0: 
                self.Positive_test.append(self.Positive[indexdev])
                self.Negtive_test.append(self.Negtive[indexdev])
            else:
                self.Positive_dev.append(self.Positive[indexdev])
                self.Negtive_dev.append(self.Negtive[indexdev])

        # test
        for indextest in testIndex:
            self.Positive_test.append(self.Positive[indextest])
            self.Negtive_test.append(self.Negtive[indextest])

    # Save dataset after split
    def saveSplitTrainAndDev(self,savedir): 
        cwT = CSVWriter(savedir+'train_ori.csv')
        cwD = CSVWriter(savedir+ 'dev_ori.csv')
        tmpT = []
        tmpT.append(['input_sym', 'standard_sym', 'score'])
        tmpD = []
        tmpD.append(['input_sym', 'standard_sym', 'score'])
        
        for i in range(0,len(self.Positive_train)):
            tmpT.append(self.Positive_train[i])
            for item in self.Negtive_train[i]:
                tmpT.append(item)
        cwT.writeData(tmpT)

        for i in range(0,len(self.Positive_dev)):
            tmpD.append(self.Positive_dev[i])
            for item in self.Negtive_dev[i]:
                tmpD.append(item)
        cwD.writeData(tmpD)

    # load train and dev
    def loadTrainAndDev(self, trainpath, devpath):
        
        
        # train
        # clean
        self.Positive_train.clear()
        self.Negtive_train.clear()
        # parsing
        data = CSVReader(trainpath).getData()[1:]
        neglist = list()
        for index in range(len(data)-1, -1, -1): 
            try:
                score = float(data[index][2].strip())
            except:
                print(str(data[index])+':'+str(index))
                continue
            if math.isclose(score,1.0):
                self.Positive_train.append([data[index][0],data[index][1],score])                
                self.Negtive_train.append(sorted(neglist, key = lambda x:x[2],reverse=True))
                neglist.clear()
            else:
                neglist.append([data[index][0],data[index][1],score])
        if len(self.Negtive_train) == len(self.Positive_train) and len(self.Positive_train) > 0:
            print("Good load for train.")
        else:
            print("Error on train mapping.")

        # dev
        # clean        
        self.Positive_dev.clear()
        self.Negtive_dev.clear()
        # parsing
        data = CSVReader(devpath).getData()[1:] 
        neglist = list()
        for index in range(len(data)-1, -1, -1): 
            try:
                score = float(data[index][2].strip())
            except:
                print(str(data[index])+':'+str(index))
                continue
            if math.isclose(score,1.0):         
                self.Positive_dev.append([data[index][0],data[index][1],score])                
                self.Negtive_dev.append(sorted(neglist, key = lambda x:x[2],reverse=True))
                neglist.clear()
            else:
                neglist.append([data[index][0],data[index][1],score])
        if len(self.Positive_dev) == len(self.Negtive_dev) and len(self.Positive_dev) > 0:
            print("Good load for dev.")
        else:
            print("Error on dev mapping.")

    #Nagetive sampling: RS    
    def sampleForNeg(self,mode='train'): 

        if mode == self.TRAIN_MODE: 
            self.SamplingNegtive_train.clear()
            for neglist in self.Negtive_train:
                randomSam = random.sample(neglist, 10)
                self.SamplingNegtive_train.append(randomSam)
        if mode == self.DEV_MODE: 
            self.SamplingNegtive_dev.clear()
            for neglist in self.Negtive_dev:
                randomSam = random.sample(neglist, 10)
                self.SamplingNegtive_dev.append(randomSam)
    
    ##Nagetive sampling: DSS    
    def sampleForNeg_R1(self, mode='train'): 
        if mode == self.TRAIN_MODE: 
            self.SamplingNegtive_train.clear()
            for neglist in self.Negtive_train:
                top20p = int(len(neglist)*0.2)
                mid40p = int(len(neglist)*0.6)
                if top20p > 5:
                    randomSamTop = random.sample(neglist[:top20p], 5)
                    randomSamMid = random.sample(neglist[top20p:mid40p], 3)
                    randomSamEnd = random.sample(neglist[mid40p:], 2)
                    self.SamplingNegtive_train.append(randomSamTop+randomSamMid+randomSamEnd)
                else:
                    randomSamTop = random.sample(neglist[:top20p], int(0.5*top20p))
                    randomSamMid = random.sample(neglist[top20p:mid40p], int(0.3*top20p))
                    randomSamEnd = random.sample(neglist[mid40p:], 1)
                    self.SamplingNegtive_train.append(randomSamTop+randomSamMid+randomSamEnd)
        if mode == self.DEV_MODE: 
            self.SamplingNegtive_dev.clear()
            for neglist in self.Negtive_dev:
                top20p = int(len(neglist)*0.2)
                mid40p = int(len(neglist)*0.6)
                if top20p > 5:
                    randomSamTop = random.sample(neglist[:top20p], 5)
                    randomSamMid = random.sample(neglist[top20p:mid40p], 3)
                    randomSamEnd = random.sample(neglist[mid40p:], 2)
                    self.SamplingNegtive_dev.append(randomSamTop+randomSamMid+randomSamEnd)
                else:
                    randomSamTop = random.sample(neglist[:top20p], int(0.5*top20p))
                    randomSamMid = random.sample(neglist[top20p:mid40p], int(0.3*top20p))
                    randomSamEnd = random.sample(neglist[mid40p:], 1)
                    self.SamplingNegtive_dev.append(randomSamTop+randomSamMid+randomSamEnd)

    #padding
    def padding(self, raw_txt, maxlen):
        result = []
        raw_txt = raw_txt.replace('.', '').replace(' ', '').replace('\n', '')
        
        for char in raw_txt:
            if char not in self.word2id:
                 char='<UNK>'
            result.append(self.word2id[char])
        result += [0] * (maxlen - len(result))
        
        return result
    
    # generate dataset for torch.dataset
    def generateDataset(self, label2idpath, x1_maxlen, x2_maxlen, mode='train'):
        # load word2id label2id
        label2id = json.load(open(label2idpath, encoding='utf-8'))
        
        x1_list, x2_list, label_list, true_label_list = [], [], [], []

        if mode == self.TRAIN_MODE: 
            positive = self.Positive_train
            negtiveLiset = self.SamplingNegtive_train
        if mode == self.DEV_MODE: 
            positive = self.Positive_dev
            negtiveLiset = self.SamplingNegtive_dev
        
        self.dataset_gen = None
        self.iter_dateset_gen =None

        for index in range(0,len(positive)):
            posSample = positive[index]
            negSampleList = negtiveLiset[index]
            if posSample[1] not in self.zz2label:
                continue
            x1_list.append(self.padding(posSample[0], x1_maxlen))
            x2_list.append(self.padding(posSample[1], x2_maxlen))
            label_list.append([label2id[self.zz2label[posSample[1]]]])
            true_label_list.append(posSample[2])
            for negSample in negSampleList:
                if negSample[1] not in self.zz2label:
                    continue
                x1_list.append(self.padding(negSample[0], x1_maxlen))
                x2_list.append(self.padding(negSample[1], x2_maxlen))
                label_list.append([label2id[self.zz2label[negSample[1]]]])
                true_label_list.append(negSample[2])

        #covert to numpy
        x1_np = np.array(x1_list)
        x2_np = np.array(x2_list)
        labels_np = np.array(label_list)
        y_np = np.array(true_label_list)

        #covert to torch
        x1 = torch.from_numpy(x1_np)
        x2 = torch.from_numpy(x2_np)
        labels = torch.from_numpy(labels_np)
        y = torch.from_numpy(y_np)

        #build dataset
        self.dataset_gen = torch.utils.data.TensorDataset(x1, x2, labels, y) 
        self.iter_dateset_gen = torch.utils.data.DataLoader(self.dataset_gen , batch_size=self.batch_size, shuffle=True)
        return self.iter_dateset_gen
    
    # output test
    def generateTest(self, savepath):
        prediction = []

        for posSample in self.Positive_test:
            prediction.append(posSample[0]+'\t'+self.zz2label[posSample[1]]+'\t'+posSample[1]+'\n')

        with open(savepath+'predict.txt','w', encoding='utf-8') as f:
            f.writelines(prediction)
    
    # generate mapping files
    def generateIdMapping(self, savepath):
        #bz2id.json
        bz2id = dict()
        count = 0
        for tmp in list(self.zz2label.keys()):
            bz2id[tmp] = count
            count += 1
        json.dump(bz2id, open(savepath+"bz2id.json", 'w', encoding='utf-8'), indent=4)

        #bz2vec.npy
        bz2id = json.load(open(savepath+"bz2id.json", encoding='utf-8'))
        bz2vec = []
        for k, v in bz2id.items():
            vec = self.padding(k, 16)
            bz2vec.append(vec)
            # print(bz2vec[v])
        np.save(savepath+"bz2vec.npy", bz2vec)

        #label_bz2id.json
        data = ExcelReader(self.standardPath).getSheetData('Sheet1')[1:]

        label_bz2id = dict()
        for line in data:
            if line[3] == 'None' or line[1] == 'None':
                continue

            if line[3] not in bz2id:
                continue

            if line[1] not in label_bz2id:
                label_bz2id[line[1]] = [bz2id[line[3]]]
            else:
                label_bz2id[line[1]].append(bz2id[line[3]])

        json.dump(label_bz2id, open(savepath+"label_bz2id.json", 'w', encoding='utf-8'), indent=4)