#!/usr/bin/env python
# coding: utf-8

# In[7]:


#! python3
# -*- encoding: utf-8 -*-
'''
@Time    :   2023/05/15 
@Author  :   Jin Yujun
@Version :   2.0
@Contact :   13738026651@163.com
'''

#微调Bert预训练模型对tweet文本进行分类
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import BertModel
#from transformers import BertForSequenceClassification
import os
import wandb
import torch.optim as optim
from tqdm import tqdm


# os.environ["WANDB_API_KEY"]='5598353c669b438b701d00322e34366480ed4afe'
# os.environ["WANDB_MODE"] = "offline"
# config =dict (
#   epochs= 10,
#   learning_rate= 5e-5,
#   batch_size= 256
# )

# sweep_configuration = {
#     'method': 'grid',
#     'name': 'sweep',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'val_acc'
#         },
#     'parameters': {
#         'lr': {'values': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]}
#      }
# }

# wandb.init(project="pytorch",
#           name="Disas_text",
#           config = config,
#           )

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="pytorch")

# wandb.agent(sweep_id)

CUDA=torch.cuda.is_available()
devices  = torch.device('cuda' if CUDA else 'cpu')

pt_save_directory = './model'

data_tensor = np.load('sentences_tokenizer.npy',allow_pickle=True)
for key in data_tensor.item():
    data_tensor.item()[key] = data_tensor.item()[key].to(devices)

#data_tensor['input_ids'] = data_tensor['input_ids'].to(devices)
#data_tensor['attention_mask'] = data_tensor['attention_mask'].to(devices)
#data_tensor['labels'] = data_tensor['labels'].to(devices)

class DisasterDataset(Dataset):
    def __init__(self,data):
        self.data = data.item()

    def __getitem__(self, idx):
        labels = self.data['labels'][idx]
        sentences = {}
        sentences['input_ids'] = self.data['input_ids'][idx]
        sentences['attention_mask'] = self.data['attention_mask'][idx]
        
        return sentences, labels
    def __len__(self):
        return len(self.data['input_ids'])

    
# # 2. 这里开始利用huggingface搭建网络模型
# # 这个类继承再nn.module,后续再详细介绍这个模块
# # 
class BertClassificationModel(nn.Module):
    def __init__(self,dropout,hidden_size=768):
        super(BertClassificationModel, self).__init__()
        # 这里用了一个简化版本的bert
        model_name = 'bert-base-uncased'


        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
#        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=2)
#        for p in self.bert.parameters(): # 冻结bert参数
#                p.requires_grad = False
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(hidden_size,2)

    def forward(self, batch_sentences):   # [batch_size,1]
        
        input_ids = batch_sentences['input_ids']
        attention_mask = batch_sentences['attention_mask']
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask) # 模型
        last_hidden_state =bert_out[0] # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state=last_hidden_state[:,0,:] # 变量
        self.dropout(bert_cls_hidden_state)
        fc_out=self.fc(bert_cls_hidden_state) # 模型
        return fc_out                         


# In[10]:


def main():
    batchsize = 256  # 定义每次放多少个数据参加训练
    
    Datas = DisasterDataset(data_tensor) # 加载训练集
#    train_data, test_data = torch.utils.data.random_split(Datas,[6144,1469])
    train_loader = torch.utils.data.DataLoader(Datas, batch_size=batchsize, shuffle=False)#遍历train_dataloader 每次返回batch_size条数据
#    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=False)

    # 这里搭建训练循环，输出训练结果
    epoch_num = 3  
    dropout = 0.3
    # 初始化模型
    model=BertClassificationModel(dropout=dropout)
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model = nn.DataParallel(model).to(devices)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.4 )

    criterion = nn.CrossEntropyLoss()
    criterion.to(devices)
    
    print("模型数据已经加载完成,现在开始模型训练。")
    for epoch in tqdm(range(epoch_num)):
        model.train()
        acc = 0
        for i, (data,labels) in enumerate(train_loader, 0):

            output = model(data)
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels)  # 计算误差
            out = output.argmax(dim=1)
            acc += (out == labels).sum().item()
            
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        scheduler.step()
            # 打印一下每一次数据扔进去学习的进展
#            print('batch:%d loss:%.5f' % (i, loss.item()))
        
        # 打印一下每个epoch的深度学习的进展i
        train_acc = acc / 7613
#         wandb.log({'accuracy': train_acc, 'loss': loss.item()})
        print('epoch:%d acc:%.5f' % (epoch, train_acc))
    
    torch.save(model.state_dict(),'model.pth')
#         num = 0
#     model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用；

#     for j, (data,labels) in enumerate(test_loader, 0):

#         output = model(data)
#         out = output.argmax(dim=1)
#         num += (out == labels).sum().item()
#         val_acc = num / 1469
#         print('val_Accuracy:', val_acc)

#         wandb.log({'val_acc': val_acc})


if __name__ == '__main__':
    main()

# In[ ]:




