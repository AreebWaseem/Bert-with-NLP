#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:39:48 2019

@author: areebwaseem
"""



import pandas as pd 
import numpy as np 
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('subtaskA_data_all.csv')

labels = pd.read_csv('subtaskA_answers_all.csv',header=None)


data = data.values.tolist()
labels = labels.values.tolist()

train_data = []
train_labels = []

for index in range(len(data) - 1):
    
    label = labels[index][1]
    sentence_one = data[index][1]
    sentence_two = data[index][2]
    if(label == 0):
        train_data.append(sentence_one)
        train_labels.append(0)
        train_data.append(sentence_two)
        train_labels.append(1)
    else:
        train_data.append(sentence_one)
        train_labels.append(1)
        train_data.append(sentence_two)
        train_labels.append(0)
        
    


org_len = len(train_data)

slicing = int(len(train_data) * 0.01)


test_data = train_data[slicing:org_len ]
test_labels = train_labels[slicing:org_len]

train_data = train_data[0:slicing]
train_labels = train_labels[0: slicing]

    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tr_tokenized = []
tst_tokenized = []

for sentence in train_data:
    tr_tokenized.append(['[CLS]'] + tokenizer.tokenize(sentence)[:480])
 

for sentence in test_data:
    tst_tokenized.append(['[CLS]'] + tokenizer.tokenize(sentence)[:480]) 
    



tr_tok_ids = []
tst_tok_ids = []

for toks in tr_tokenized:
    tr_tok_ids.append(tokenizer.convert_tokens_to_ids(toks))
    
for toks in tst_tokenized:
    tst_tok_ids.append(tokenizer.convert_tokens_to_ids(toks))
    



tr_tok_ids = pad_sequences(tr_tok_ids, maxlen=480, truncating="post", padding="post", dtype="int")

tst_tok_ids = pad_sequences(tst_tok_ids, maxlen=480, truncating="post", padding="post", dtype="int")



train_y = np.array(train_labels) == 1

test_y = np.array(test_labels) == 1


    
tr_masks = []

tst_masks = []

for tok_id in tr_tok_ids:
    ind_tok = []
    for elem in tok_id:
        if(elem > 0):
            ind_tok.append(1.0)
        else:
            ind_tok.append(0.0)
     
    tr_masks.append(ind_tok)
    
    
    
for tok_id in tst_tok_ids:
    
    ind_tok = []
    
    for elem in tok_id:
        if(elem > 0):
            ind_tok.append(1.0)
        else:
            ind_tok.append(0.0)
     
    tst_masks.append(ind_tok)
    
    


tr_masks_tensor = torch.tensor(tr_masks)

tst_masks_tensor = torch.tensor(tst_masks)


train_tokens_tensor = torch.tensor(tr_tok_ids)

train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()

test_tokens_tensor = torch.tensor(tst_tok_ids)

test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()


tr_dataset =  torch.utils.data.TensorDataset(train_tokens_tensor, tr_masks_tensor , train_y_tensor)


tr_dataloader =  torch.utils.data.DataLoader(tr_dataset)


tst_dataset =  torch.utils.data.TensorDataset(test_tokens_tensor, tst_masks_tensor, test_y_tensor)


tst_dataloader =  torch.utils.data.DataLoader(tst_dataset)


class BertBinaryClassifier(nn.Module):
    
    def __init__(self, dropout=0.1):
        
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


bert = BertBinaryClassifier()


optimizer = torch.optim.Adam(bert.parameters(), lr=3e-6)


bert.train()


for step_num, batch_data in enumerate(tr_dataloader):
    
    token_ids, masks, labels = tuple(t for t in batch_data)
    
    probas = bert(token_ids, masks)
    
    loss_func = nn.BCELoss()
    
    total_loss = loss_func(probas, labels)
    
    bert.zero_grad()
    
    total_loss.backward()
    
    optimizer.step()
    
    print('iteration complete')



bert.eval()

bert_predictions = []


with torch.no_grad():
    
    for step_num, batch_data in enumerate(tst_dataloader):
        
        token_ids, masks, labels = tuple(t for t in batch_data)
        
        logits = bert(token_ids, masks)
        
        loss_function = nn.BCELoss()
        
        loss = loss_function(logits, labels)
        
        numpy_logits = logits.cpu().detach().numpy()
        
        bert_predictions += list(numpy_logits[:, 0] > 0.5)
        
        print('iteration complete')
        
        
tot_count = 0

for index in range(len(test_y) -1):
    if(test_y[index] == bert_predictions[0]):
        tot_count = tot_count + 1
    

    
print('Accuracy: ', float(float(tot_count)/float(len(test_y))))


# Accuract 0.50
