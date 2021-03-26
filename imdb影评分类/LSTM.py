'''
Date: 2021-03-25 16:16:06
LastEditors: ELROY
LastEditTime: 2021-03-25 16:43:45
FilePath: \torch\LSTM.py
'''
import torch
import torch.nn as nn

batch_size = 10
seq_len = 20
embedding_dim  = 30
word_vocab = 100
hidden_size = 18
num_layer = 1


class My_LSTM(nn.Module):

    def __init__(self):
        super(My_LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,batch_first=True,bidirectional=True)
        self.embedding = nn.Embedding(word_vocab,embedding_dim)
    
    def forward(self,input):
        input_embeded = self.embedding(input)
        output,(h_n,c_n) = self.lstm(input_embeded)
        return output,(h_n,c_n)



if __name__=='__main__':
    model = My_LSTM()
    input = torch.randint(low=2,high=100,size=(batch_size,seq_len))
    out,(h_2,c_2) = model(input)
    print(out.size())
    print('*'*100)
    print(h_2.size())
    print('*'*100)
    print(c_2.size())