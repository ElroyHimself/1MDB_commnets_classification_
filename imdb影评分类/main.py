'''
Date: 2021-03-24 20:46:22
LastEditors: ELROY
LastEditTime: 2021-03-24 20:54:59
FilePath: \torch\main.py
'''
from word_sequence import word2sequence
import pickle
import os
from dataset import tokenize
from tqdm import tqdm #打印进度条

if __name__=='__main__':

    
    ws = word2sequence()
    path = "IMDB/aclImdb/train"
    temp_data_path = [os.path.join(path,'pos'),os.path.join(path,'neg')]
    for data_path in temp_data_path:
        file_paths =[ os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith('txt')]
        for file_path in tqdm(file_paths):
            sentence = tokenize(open(file_path,errors='ignore').read())
            ws.fit(sentence)
    ws.build_vocab(min=10)
    #ws.build_vocab(min=10,max_features=10000)
    pickle.dump(ws,open('./model/ws.pkl','wb'))
    print(len(ws))