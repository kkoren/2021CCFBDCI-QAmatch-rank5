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

class QuestionMatching_base(nn.Layer):
    '''
    base模型
    dropout改为0.15
    '''
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.15)

        #线性变换层,Out=XW+b
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)


    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):

        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)
    

        kl_loss = 0.0

        return logits1, kl_loss

class QuestionMatching_BN(nn.Layer):
    '''
    base模型+BN
    dropout改为0.15
    '''
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.15)



        self.linear=nn.Linear(self.ptm.config["hidden_size"], self.ptm.config["max_position_embeddings"])
        self.batchnorm1d=nn.BatchNorm1D(self.ptm.config["max_position_embeddings"])
        self.relu=nn.ReLU()

        # self.relu=nn.GELU()
        #线性变换层,Out=XW+b
        self.classifier = nn.Linear(self.ptm.config["max_position_embeddings"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        
        #pretrained_model返回
        #sequence_output[batch_size, sequence_length, hidden_size],
        #pooled_output [batch_size, hidden_size]   The output of first token ([CLS]) in sequence
        _, cls_embedding1 = self.ptm(input_ids, 
                                     token_type_ids, #用于区分当前token是属于哪个句子的
                                     position_ids,#明确每个token是在什么位置上，从0到最后依次编号。
                                     attention_mask,#指定对哪些词进行self-Attention操作(padding的位置是不需要参与attention计算的)
                                     )
        
        cls_embedding1 = self.dropout(cls_embedding1)
        cls_embedding1=self.linear(cls_embedding1)
        cls_embedding1=self.batchnorm1d(cls_embedding1)
        cls_embedding1=self.relu(cls_embedding1)
        cls_embedding1=self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)
        

        kl_loss = 0.0

        return logits1, kl_loss

