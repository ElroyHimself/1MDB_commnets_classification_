'''
Date: 2021-03-24 19:23:07
LastEditors: ELROY
LastEditTime: 2021-03-25 20:05:47
FilePath: \torch\word_sequence.py
'''
import torch
import numpy as np

class word2sequence():

    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    
    UNK = 0
    PAD = 1 

    def __init__(self):
        self.dict = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD
        }
        self.count = {}

    def fit(self,sentence):
        """把单个句子保存到dict中
            sentence[word1,word2,word3,...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1


    def build_vocab(self,min=5,max=None,max_features=None):
        """
        生成词典
        min：最小出现的次数
        max：最大出现的次数
        max_features:一共保留多少个词语
        return
        """
        #删除count中词频小于min的词
        if min is not None:
            self.count = {word:value for word,value in self.count.items() if value >min  }

        #删除词频大于max的词
        if max is not None:
           self.count = {word:value for word,value in self.count.items() if value <max }
        
        #限制保留的词语数
        if max_features is not None:
            temp=sorted(self.count.items(),key = lambda x : x[-1],reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)#新加入词的ID 等于原来的词的个数加1

        #键和值反转 ID2WORD 
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None):
        """
        句子转化为序列
        sentence[word1,word2,....]
        """
        
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            
            if max_len < len(sentence):
                sentence = sentence[:max_len]

        return [self.dict.get(word,self.UNK) for word in sentence]
        """for index,word in enumerate(sentence):
            r[index] = self.to_index(word)"""

        #return np.array(r,dtype=np.int64)

    def inverse_transform(self,indices):
        return[self.inverse_dict.get(idx) for idx in indices ]

    def __len__(self):
        return len(self.dict)


if __name__=='__main__':
    """ws = word2sequence()
    ws.fit(['who','i','am'])
    ws.fit(['who','are','you'])
    ws.build_vocab(min=0)
    print(ws.dict)
    
    ret = ws.transform(['who','the','fuck','you','are'],max_len=10)
    print(ret)"""

    
    from word_sequence import word2sequence
    import pickle
    import os
    from dataset import tokenize

    ws = word2sequence()
    data_path = "IMDB/aclImdb/train"
    temp_data_path = [os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
    for data_path in temp_data_path:
        file_name = os.listdir(data_path)
        file_path = os.path.join(data_path,file_name)
        sentence = tokenize(open(file_path).read())
        ws.fit(sentence)
    ws.build_vocab(min=10)
    pickle.dump(ws,open('./model/ws.pkl','rb'))
    print(len(ws))