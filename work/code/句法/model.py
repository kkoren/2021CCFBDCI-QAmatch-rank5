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
import gnn
from data import index_sample
class QuestionMatching_parser(nn.Layer):
    '''
    base模型
    dropout改为0.15
    '''
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.15)
        self.gat=gnn.GAT(input_size=768, hidden_size=384, output_size=192, dropout=0.0, alpha=0.1, heads=12, layer=2)#增加GAT网络
        #线性变换层,Out=XW+b
        self.classifier = nn.Linear(self.ptm.config["hidden_size"]+384, 2)
    

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                adjacency_matrix=None,
                head_ids=None,
                do_evaluate=False):

        #sequence_out[128,68,768] pool_out[128,768]
        sequence_out, pool_out = self.ptm(input_ids, token_type_ids, position_ids,attention_mask)
        #adjacency_matrix[128,68,68]
        #head_ids[128,2]

        # 将ernie的表示和邻接矩阵输入到gat网络中得到包含句子结构信息的表示
        gat_emb = self.gat(sequence_out, adjacency_matrix)#[128,68,192]

        # 提取核心词的表示
        gat_emb = index_sample(gat_emb, head_ids)#[128,192*2]


        # 将[CLS]emb和核心词的表示拼接，供下游网络使用
        cls_feats = paddle.concat([pool_out, gat_emb], axis=1)#[128,68,768+192]

        # cls_feats=paddle.mean(cls_feats,axis=1)

        cls_feats = self.dropout(cls_feats)
        logits1=self.classifier(cls_feats)

        return logits1

