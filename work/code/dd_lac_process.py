# -*- coding: utf-8 -*-
"""
Created on 2021/11/07

@author: Christina
"""

import sys
import json
import pickle
from tokenization import BasicTokenizer
import sys
import json
import pickle
from tokenization import BasicTokenizer

import pandas as pd
from ddparser import DDParser
from LAC import LAC
import pandas as pd

'''
用于将原文本拥有词性和句法结构
'''

#%%In[1] 读取数据
def read_csvToDF(data_path, is_test=False):
    """
    Parameters
    ----------
    data_path : str
        文件路径.
    is_test : bool, optional
        是否是测试集. The default is False.
        
    读取文件使用
    Returns
        返回dataframe类型数据集

    """
    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_data=[i for i in line.rstrip().split("\t")]
            if is_test == False:
                if len(tmp_data) !=3:
                    continue
                data.append(tmp_data)
            else:
                if len(tmp_data) != 2:
                    continue
                data.append(tmp_data)
                
    if is_test == False:
        df=pd.DataFrame(data,columns=['query1','query2'])
        #调整label类型为int
        df['label']= df['label'].astype(int)
        return df
    else:
        return pd.DataFrame(data,columns=['query1','query2','label'])

#读取test_B数据集
train=read_csvToDF(data_path='test_B.tsv',is_test=True)

#%%In[2] 处理数据
# 装载LAC模型
lac = LAC(mode='lac')
# 加载分词函数
tokenizer = BasicTokenizer()
# 加载句法分析工具
ddp = DDParser(use_cuda=True,buckets=True,batch_size=1000)

#%%In[3] 
# 得到句法结果
train['ddp_res_b'] = [str(ddp_res) for ddp_res in ddp.parse([tokenizer._clean_text(query) for query in train['query2'].tolist()])]
train['ddp_res_a'] = [str(ddp_res) for ddp_res in ddp.parse([tokenizer._clean_text(query) for query in train['query1'].tolist()])]
# 得到词性结果
train['lac_word_a']=train['query1'].apply(lambda x:lac.run(x))
train['lac_word_b']=train['query2'].apply(lambda x:lac.run(x))

# 将分词结果分为不同列
train['lca_word1']=train['lac_word_a'].apply(lambda x:x[0])
train['lca_a']=train['lac_word_a'].apply(lambda x:x[1])
train['lca_word2']=train['lac_word_b'].apply(lambda x:x[0])
train['lca_b']=train['lac_word_b'].apply(lambda x:x[1])

#%%In[3] 保存
train[['query1','query2','ddp_res_a','ddp_res_b','lca_word1','lca_a','lca_word2','lca_b']].to_csv('test_B.txt', sep='\t',header=None, index=False)

