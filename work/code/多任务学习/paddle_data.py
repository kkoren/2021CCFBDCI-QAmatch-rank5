# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christian
"""

import paddle
import numpy as np
import pandas as pd
from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,mode='train', batch_size=1,batchify_fn=None,trans_fn=None,shuffle =False):
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


#读取训练集,返回生成器
def read_text_pair(data_path, is_test=False):
    """Reads data."""    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")

            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}

def read_dataset_mt(data_path, is_test=False,dataset_id=0):
    """Reads data."""   
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'dataset_id':dataset_id,'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'dataset_id':dataset_id,'query1': data[0], 'query2': data[1]}
 

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
def convert_query_to_ids_datasetid(example,tokenizer,max_seq_length=62, is_test=False):

    query1, query2 = example["query1"], example["query2"]
    dataset_id=example['dataset_id']
    #拼接tokenizer
    encoded_inputs = tokenizer(text=query1, text_pair=query2, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]#字在词典中id
    token_type_ids = encoded_inputs["token_type_ids"]#用来区分第一句还是第二句

    if not is_test:#训练集，label转int
        label = np.array([example["label"]], dtype="int64")
        return dataset_id,input_ids, token_type_ids, label
    else:
        return dataset_id,input_ids, token_type_ids



def convert_query_to_ids(example,tokenizer,max_seq_length=62, is_test=False):

    query1, query2 = example["query1"], example["query2"]

    #拼接tokenizer
    encoded_inputs = tokenizer(text=query1, text_pair=query2, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]#字在词典中id
    token_type_ids = encoded_inputs["token_type_ids"]#用来区分第一句还是第二句

    if not is_test:#训练集，label转int
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids



