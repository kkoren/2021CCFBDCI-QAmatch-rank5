import sys
import json
import pickle
from tokenization import BasicTokenizer

import pandas as pd
from ddparser import DDParser
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

test=read_csvToDF(data_path='test_A.tsv',is_test=True)
tokenizer = BasicTokenizer()
ddp = DDParser(use_cuda=True,buckets=True,batch_size=1000)

test['ddp_res_a'] = [str(ddp_res) for ddp_res in ddp.parse([tokenizer._clean_text(query) for query in test['query1'].tolist()])]
test['ddp_res_b'] = [str(ddp_res) for ddp_res in ddp.parse([tokenizer._clean_text(query) for query in test['query2'].tolist()])]

#保存
test[['query1','query2','ddp_res_a','ddp_res_b']].to_csv('test.txt', sep='\t',header=None, index=False)