# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christian
"""

import paddle
import numpy as np
import pandas as pd
from paddlenlp.datasets import MapDataset
from graph import *
from paddle.fluid import layers


def create_dataloader(dataset,mode='train', batch_size=1,batchify_fn=None,trans_fn=None):
    if trans_fn:#将数据集转为ids
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False#训练集打乱
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

#读取训练集
def read_text_pair(data_path, is_test=False):
    """Reads data."""    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 5:
                    continue
                yield {'query1': data[0], 'query2': data[1],'ddp_res_a':data[2],'ddp_res_b':data[3] ,'label': data[4]}
            else:
                if len(data) != 4:
                    continue
                yield {'query1': data[0], 'query2': data[1],'ddp_res_a':data[2],'ddp_res_b':data[3] }

#读取训练集变df
def read_csvToDF(data_path, is_test=False):
    data = []
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_data=[i for i in line.rstrip().split("\t")]
            if is_test == False:
                if len(tmp_data) != 3:
                    continue
                data.append(tmp_data)
            else:
                if len(tmp_data) != 2:
                    continue
                data.append(tmp_data)
                
    if is_test == False:
        df=pd.DataFrame(data,columns=['query1','query2','label'])
        #调整label类型为int
        df['label']= df['label'].astype(int)
        return df
    else:
        return pd.DataFrame(data,columns=['query1','query2'])
    
#把文字转成数字 ids 类型
def convert_query_to_ids(example, tokenizer, max_seq_length=62, is_test=False):

    query1, query2 = example["query1"], example["query2"]
    
    #拼接tokenizer
    encoded_inputs = tokenizer(text=query1, text_pair=query2, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]#字在词典中id
    token_type_ids = encoded_inputs["token_type_ids"]#用来区分第一句还是第二句


    ddp_res_a = eval(example['ddp_res_a'])#变成字典
    ddp_res_b = eval(example['ddp_res_b'])

    if type(ddp_res_a)==str: 
        ddp_res_a = eval(ddp_res_a)#变成字典
    
    if type(ddp_res_b)==str:
        ddp_res_b = eval(ddp_res_b)#变成字典

    query1_token=tokenizer.tokenize(query1)
    query2_token=tokenizer.tokenize(query2)

    # 获取句子a的弧及核心词索引
    # (arc_tails: arc_tails表示映射后所有弧尾的索引,arc_heads: arc_heads表示映射后所有弧头的索引)
    # head_id: 表示映射后核心词的所有,只有一个
    arcs_a, head_id_a = get_arcs_and_head_in_wordpiece(ddp_res_a, query1_token)
    # 获取句子b的弧及核心词索引
    arcs_b, head_id_b = get_arcs_and_head_in_wordpiece(ddp_res_b, query2_token)


    # 获取句子a和b组成的邻接矩阵(max_len_a + max_len_b + 3,max_len_a + max_len_b + 3)
    adjacency_matrix = get_adj_of_two_sent_in_ernie(arcs_a,int(token_type_ids.count(0)-2), arcs_b,int(token_type_ids.count(1)-1) )
    
    # 获取映射后核心词索引,tuple,加了cls和sep的位置 获取新的核心词位置
    head_ids = transfor_head_id_for_ernie(head_id_a, int(token_type_ids.count(0)-2), head_id_b, int(token_type_ids.count(1)-1))

    if not is_test:#训练集，label转int
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids,adjacency_matrix,head_ids, label
    else:
        return input_ids, token_type_ids,adjacency_matrix,head_ids


def index_sample(x, index):
    '''
    x=将ernie的表示和邻接矩阵输入到gat网络中得到包含句子结构信息的表示[batch_size,seq_len,50]
    index=head_ids[batch_size,2]
    '''
    x_s = x.shape#[batch_size,seq_len,50]
    dim = len(index.shape) - 1
    assert x_s[:dim] == index.shape[:dim]#检查维度是否相等
    
    r_x = layers.reshape(x, shape=(-1, *x_s[dim:]))
    index = layers.reshape(index, shape=(index.shape[0], index.shape[1], 1))
    # generate arange index, shape like index
    # arr_index = layers.arange(start=0, end=layers.cast(layers.shape(x)[0], ), dtype=index.dtype)
    batch_size = layers.cast(layers.shape(index)[0], dtype=index.dtype)
    zero = layers.fill_constant(shape=[1], dtype=index.dtype, value=0)
    one = layers.fill_constant(shape=[1], dtype=index.dtype, value=1)
    arr_index = layers.unsqueeze(layers.range(zero, batch_size, one, dtype=index.dtype), [1, 2])

    arr_index = layers.expand_as(arr_index, index)
    #  genrate new index
    new_index = layers.concat([arr_index, index], -1)
    new_index = layers.reshape(new_index, (-1, 2))
    # get output
    out = layers.gather_nd(r_x, new_index)
    out = layers.reshape(out, (-1, x_s[-1] * 2))
    return out