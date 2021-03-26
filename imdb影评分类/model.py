'''
Date: 2021-03-25 14:33:06
LastEditors: ELROY
LastEditTime: 2021-03-25 18:40:37
FilePath: \torch\model.py
'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.modules import dropout
#from dataset import ws
from lib import ws,max_len


class MyModel(nn.Module):
    
    def __init__(self):
        super(MyModel,self).__init__()
        self.hidden_size = 128
        self.embedding_dim = 100
        self.num_layer = 2
        self.bidirectional = True
        self.bi_num = 2 if self.bidirectional else 1
        self.dropout = 0.5
        #以上为超参数

        self.embedding = nn.Embedding(len(ws),self.embedding_dim)#[N,300]
        self.lstm = nn.LSTM(input_size=100,hidden_size=self.hidden_size,num_layers=self.num_layer,batch_first=True,bidirectional=True,dropout=self.dropout)
        self.fc1 = nn.Linear(self.hidden_size*2,2)
        #self.fc2 = nn.Linear(20,2)


    def forward(self,input):
        """
        输入[batch_size,max_len][128,20]
        
        """
        x = self.embedding(input)#经过embedding后[128,20,100]
        #x:[batch_size,max_len,hidden_size*2],h_n[2*2,batch_Size,hidden_size],c_n[2*2,batch_Size,hidden_size]
        x,(h_n,c_n) =self.lstm(x)#[128,20,256]
        #双向LSTM 获取两个方向最后一次的output，进行concatenate

        output_fw = h_n[-2,:,:]#[20,128]
        output_bw = h_n[-1,:,:]#[20,128]
        output = torch.cat([output_fw,output_bw],dim=-1)#[20,256]

        #x =x.view(128,-1)#[128,5120]
        out = self.fc1(output)#[20,2]
        #out = self.fc2(out)
        #out =self.fc2(out)
        return F.log_softmax(out,dim=-1)