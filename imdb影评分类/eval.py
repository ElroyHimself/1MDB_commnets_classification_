'''
Date: 2021-03-25 18:45:05
LastEditors: ELROY
LastEditTime: 2021-03-25 19:34:09
FilePath: \torch\eval.py
'''
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
#from dataset import ws
from lib import ws,max_len,test_batch_size
from model import MyModel
from dataset import get_dataloader
from tqdm import tqdm
import numpy as np


model = MyModel()
criterion = nn.CrossEntropyLoss()

def eval():
    loss_list = []
    acc_list = []
    data_loader=get_dataloader(train=False,batch_size=test_batch_size)
    for idx,(input,target) in tqdm(enumerate(data_loader),total=len(data_loader),ascii=True):
        with torch.no_grad():
            output = model(input)
            cur_loss = criterion(output,target)
            loss_list.append(cur_loss.numpy())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target.float().mean())
            acc_list.append(cur_acc.numpy())
    print('total loss,acc',np.mean(loss_list),np.mean(acc_list))


if __name__ =='__main__':
    eval()