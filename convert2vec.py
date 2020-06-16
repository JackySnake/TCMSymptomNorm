import json

#covert token to id
class Converter(object):
    def __init__(self):
        self.word2id = json.load(open("data/mapping/word2id_whole.json", encoding='utf-8'))

    def convert(self, raw_txt, maxlen=16):
        result = []
        for char in raw_txt:
            if char not in self.word2id:
                return []
            result.append(self.word2id[char])
        result += [0] * (maxlen - len(result))
        return result

