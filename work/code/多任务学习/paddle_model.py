# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christian
"""

"""
#BASE +BN层
#dropout改为0.15
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp


class MultitaskClassifier(nn.Layer):
    '''
    多任务学习模型
    一个数据集对应一个线性层
    '''
    def __init__(self, pretrained_model, dropout=None,labels_num_list=None):
        super(MultitaskClassifier, self).__init__()
        
        self.ptm = pretrained_model#模型
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.15)#dropout层

        self.classifier = nn.LayerList([nn.Linear(self.ptm.config['hidden_size'],2) for _ in labels_num_list])
        
        self.dataset_id = 0#初始化为0

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        
        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,attention_mask)
        
        cls_embedding1 = self.dropout(cls_embedding1)
        
        logits =self.classifier[self.dataset_id](cls_embedding1)
        
        return logits
    
    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id

class Classifier(nn.Layer):
    '''
    多任务学习模型
    一个数据集对应一个线性层
    '''
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super(Classifier, self).__init__()
        
        self.ptm = pretrained_model#模型
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.15)#dropout层
        
        self.classifier = nn.Linear(self.ptm.config['hidden_size'],2)
        self.sigmoid=nn.Sigmoid()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        
        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,attention_mask)
        
        cls_embedding1 = self.dropout(cls_embedding1)
     
        logits =self.classifier(cls_embedding1)

        # logits=self.sigmoid(logits)

        return logits
    