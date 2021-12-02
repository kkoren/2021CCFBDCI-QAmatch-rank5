# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christian
"""

import paddle
import numpy as np
import pandas as pd
from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False#训练集打乱
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

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
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}

def random_masking(token_ids,tokenizer):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))#随机数
    source, target = [], []
    #source加入mask id
    #target加入token_ids
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer.mask_token_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer.vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target

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
def convert_query_to_ids(example, tokenizer, max_seq_length=68,is_test=False,random=False):
    '''
    添加一个前缀pattern：“间接回答问题”/ “直接回答问题”，分别对应label 0 / 1，pattern的得分只需看第一个位置中“间”/“直”两个token的概率谁高即可。
    对于unlabel data，可以不增加pattern 进行mlm任务

    Parameters
    ----------
    example : TYPE
        DESCRIPTION.
    tokenizer : TYPE
        DESCRIPTION.
    max_seq_length : TYPE, optional
        DESCRIPTION. The default is 62.
    is_test : TYPE, optional
        DESCRIPTION. The default is False.
    prefix : TYPE, optional
        DESCRIPTION. The default is '很相似句子对：'.
    random : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    query1, query2= example["query1"], example["query2"]

    label= int(example["label"]) if len( example)==3 else 2
    
    
    # 对应的任务描述
    prefix = u'很相似句子对：'
    mask_idx=1 #掩盖pattern中字的位置,因为还有cls所以为1
    pos_id = tokenizer(u'很')['input_ids'][1]#取很的id
    neg_id = tokenizer(u'不')['input_ids'][1]#取不的id
        
    #若已经标注数据，加上前缀或者是测试数据
    query1 = prefix + query1
    
    #句子对输入
    encoded_inputs = tokenizer( text=query1, text_pair=query2, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]#字在词典中ids
    token_type_ids = encoded_inputs["token_type_ids"]#用来区分第一句还是第二句
    
    
    #source_ids是加了[MASK]的句子id
    #target_ids是还原后的句子id
    if random:#随机掩盖
        source_ids, target_ids = random_masking(input_ids,tokenizer)
    else:
        source_ids, target_ids = input_ids[:], input_ids[:]

    #有label的数据
    if label == 0:#不相似
        source_ids[mask_idx] = tokenizer.mask_token_id#[MASK]的id
        target_ids[mask_idx] = neg_id#修改第一个字，改为'不'的id
    elif label == 1:#很相似
        source_ids[mask_idx] = tokenizer.mask_token_id#[MASK]的id
        target_ids[mask_idx] = pos_id#修改第一个字，改为'很'的id

    elif is_test:#测试集就掩盖'很'，'不'
        source_ids[mask_idx] = tokenizer.mask_token_id#[MASK]的id

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return source_ids,token_type_ids,target_ids,label
    else:
        return source_ids,token_type_ids,target_ids
