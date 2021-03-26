'''
Date: 2021-03-25 14:42:47
LastEditors: ELROY
LastEditTime: 2021-03-25 19:21:40
FilePath: \torch\train.py
'''
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
#from dataset import ws
from lib import ws,max_len
from model import MyModel
from dataset import get_dataloader

model = MyModel()
optimizer = Adam(model.parameters(),0.001)

if os.path.exists('./model/model1.pkl'):
    model.load_state_dict(torch.load('./model/model1.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer1.pkl'))

def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        print(epoch,idx,loss.item())

        if idx % 100 == 0:
            torch.save(model.state_dict(),('./model/model1.pkl'))
            torch.save(optimizer.state_dict(),('./model/optimizer1.pkl'))
            
if __name__ == "__main__":
    for i in range(1):
        train(i)