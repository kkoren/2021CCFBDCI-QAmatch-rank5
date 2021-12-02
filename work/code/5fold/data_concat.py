import pandas as pd
from paddle_data import *


# 加载训练集
lc_train = read_csvToDF( data_path='./train/LCQMC/train', is_test=False)
lc_dev = read_csvToDF( data_path='./train/LCQMC/dev', is_test=False)
lc_test = read_csvToDF( data_path='./train/LCQMC/test', is_test=False)

lc_df=pd.concat([lc_train,lc_test,lc_dev]).dropna().reset_index(drop=True)
print(lc_df.shape)

lc_df.to_csv('lc_df.csv',header=None, index=False)