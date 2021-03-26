'''
Date: 2021-03-24 20:57:47
LastEditors: ELROY
LastEditTime: 2021-03-25 19:35:39
FilePath: \torch\lib.py
'''
import pickle
ws = pickle.load(open("./model/ws.pkl",'rb'))
max_len =20

batch_size = 512
test_batch_size = 1000