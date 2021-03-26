'''
Date: 2021-03-24 15:23:36
LastEditors: ELROY
LastEditTime: 2021-03-25 19:30:57
FilePath: \torch\dataset.py
'''
import torch
from torch.utils.data import DataLoader,Dataset
import os 
import re
from lib import ws,max_len,batch_size
import lib

data_base_path = "IMDB/aclImdb"

def tokenize(content):
    content= re.sub("<.*?>"," ",content)
    filters = ['\.','\t','\n','\x97','\x96','#','$','%','&']
    content = re.sub('|'.join(filters),'',content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self,mode):
        super(ImdbDataset,self).__init__()
        if mode == "train":
            text_path = [os.path.join(data_base_path,i) for i in ['train/neg','train/pos']]
        else:
            text_path = [os.path.join(data_base_path,i) for i in ['test/neg','test/pos']]
        self.total_file_path_list = []#所有评论文件的path
        #把所有文件名放入列表
        for i in text_path:
            self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])


    def __getitem__(self, index):
        file_path = self.total_file_path_list[index]
        #获取label
        #cur_filename = os.path.basename(file_path)
        #label_str = int(cur_filename.split('_')[-1].split(".")[0]) - 1
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == 'neg' else 1
        text = tokenize(open(file_path,errors='ignore').read().strip())
        
        return label, text

    def __len__(self):
        return len(self.total_file_path_list)




def collate_fn(batch):
    """MAX_LEN=500

    batch = list(zip(*batch))
    labels = torch.tensor(batch[0],dtype=torch.int)
    texts = batch[1]
    lengths = [len(i) if len(i)<MAX_LEN else MAX_LEN for i in texts]
    texts = torch.tensor([ws.transform(i,MAX_LEN) for i in texts])
    del batch
    return labels,texts,lengths"""
    label,content = list(zip(*batch))
    content = [ws.transform(i,max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content,label
    
def get_dataloader(train=True,batch_size=lib.batch_size):
    dataset = ImdbDataset(mode='train')
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    for idx,(text,label) in enumerate (get_dataloader):
            print(idx)
            print(text)
            print(label)
            #print(lengths)
        