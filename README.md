# TCMSymptomNorm symNormHS 
This is a traditional Chinese medicine symptom normalization method. We design and implement a text matching method a text matching model that integrates hierarchical semantic information with an attention mechanism, named it symNormHS.

This method is useful to solve the challenge that same symptoms in different literal description, one-to-many symptom description and different symptoms in similar literal description.
## Requirements
pytorch 1.2.0

## directory structure
├─checkpoints           the model checkpoint(include RS and DSS)
├─data
│  ├─dataset            train/dev/test(predicet.txt)
│  ├─mapping            the mapping file include word2vec and so on
│  │  └─HSI-Sym-id      
│  └─origindataset      the original dataset contain all positive and negative samples
│      └─sample
├─result                experiment results
│  ├─testResult-80-DSS  
│  └─testResult-80-RS
└─utils
Some data need to download from BaiduYun. The link is in the directory respectively.
## DataSet
The dataset from real-world data
```shell
data/mapping/HSI-Sym-id/bz2id.json  # normalization symptom word to id
data/mapping/HSI-Sym-id/label_bz2id.json # hierarchical semantic information to normalization symptom word id
data/mapping/HSI-Sym-id/label2id.json # hierarchical semantic information to id
data/mapping/HSI-Sym-id/bz2vec.json  # normalization symptom word to vector
data/mapping/word2id.json  # word(character in this method) to id (need to download from BaiduYun)
data/mapping/word2vec.npy  # Word2Vec(character in this method) (need to download from BaiduYun)
```

## Train
Need all data
```
python main.py
```

## Test
Need checkpoint, word2vec, mapping files
```
./test.sh
```

