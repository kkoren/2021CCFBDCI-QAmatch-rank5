import sys
sys.path.append('/home/aistudio/external-libraries')
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
from pypinyin import lazy_pinyin
import numpy as np
#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)

test=pd.read_csv("./work/raw_data/test_B_1118.tsv",sep='\t',names=['text_one','text_two'])
label=pd.read_csv("./work/user_data/tmp_result/raw_result.csv",names=['label'])
#label=pd.read_csv("C:/Users/Administrator/Desktop/lac_predict.csv",names=['label'])
test['label']=label['label']

#print(len(test[test['label']==1]))
#拼音相同
test['text_one_pinyin']=test['text_one'].apply(lambda x:lazy_pinyin(x))
test['text_two_pinyin']=test['text_two'].apply(lambda x:lazy_pinyin(x))

test_pinyin=pd.DataFrame()
test_pinyin=test[test['text_one_pinyin']==test['text_two_pinyin']]
test_pinyin=test_pinyin[test_pinyin['label']==0]
#print(test_pinyin)
#print(len(test_pinyin))
test['label_pinyin']=list(map(lambda x,y,z:1 if x==y and z==0 else z,test['text_one_pinyin'],test['text_two_pinyin'],test['label']))


import re
#正则加减乘除
test['suanshu']=test['text_one'].apply(lambda x:1 if bool(re.search(r'[\×\+\-\÷]',x)) else 0)
suansu=test[test['suanshu']==1]
#print(len(suansu[suansu['label_pinyin']==1]))
test.loc[test.suanshu==1,'label_pinyin']=0
#print(test[test['suanshu']==1])

test_A=test.copy()
from pypinyin import pinyin,Style,lazy_pinyin
test_A['pinyin1']=test_A['text_one'].apply(lambda x:pinyin(x, style=Style.TONE3, heteronym=True))
test_A['pinyin2']=test_A['text_two'].apply(lambda x:pinyin(x, style=Style.TONE3, heteronym=True))

#%%In[4]
#因为拼音拼错的错别字
import Levenshtein
test_A['dis']=-1

for index,data in enumerate(zip(test_A['pinyin1'],test_A['pinyin2'])):
    distance=0
    if len(data[0])!=len(data[1]):
        distance=-1#不属于错别字范围
    else:
        for i in range(len(data[0])):#遍历query1[[],[]]
            if len(data[0][i])==1:#取一个字的拼音，如果只有一个读音
                if data[0][i][0] not in data[1][i]:#如果该读音不在query2对应字的读音中
                    distance+=1
            else:#如果为多音字
                tmp=0
                for x in data[0][i]:#遍历多音
                    if x not in data[1][i]:#该读音不在query2对应字的读音中
                        tmp+=1
                if tmp==len(data[0][i]):#所有读音都不在query2中
                    distance+=1
    test_A.loc[index,'dis']=distance
tmp=test_A[test_A['dis']==0]
#print(len(tmp[tmp['label_pinyin']==0]))
test_A.loc[test_A.dis==0,'label_pinyin']=1
#test_A['label_pinyin'].to_csv('F:/WLL/Competitions/Compare_text/result/deal/deal_gaiic_trans_real_pool_gru_drop0.4+0.2.csv',index=None,header=None)
test_A['label_pinyin'].to_csv('./work/user_data/tmp_result/pinyin_result.csv',index=None,header=None)


import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import re
from tool import  *



#%%In[1]读取文件
def read_text_pair(data_path, is_test=False):
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
            data.append(tmp_data)
    return pd.DataFrame(data)

#读取预测文件
res=pd.read_csv('./work/user_data/tmp_result/pinyin_result.csv',names=['label'])
#rescyw=pd.read_csv('final_cyw_2245.csv',names=['label'])
#%%In[2] 读取得到词性的测试B文件
def read_csvToDF(data_path, is_test=False):
    data = []
    """读取文件"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_data=[i for i in line.rstrip().split("\t")]
            if is_test == False:
                if len(tmp_data) != 9:
                    continue
                data.append(tmp_data)
            else:
                if len(tmp_data) != 8:
                    continue
                data.append(tmp_data)
                
    if is_test == False:
        df=pd.DataFrame(data,columns=['query1','query2','ddp_res_a','ddp_res_b','lca_word1','lca_a','lca_word2','lca_b','label'])
        #调整label类型为int
        df['label']= df['label'].astype(int)
        return df
    else:
        return pd.DataFrame(data,columns=['query1','query2','ddp_res_a','ddp_res_b','lca_word1','lca_a','lca_word2','lca_b'])


test_A=read_csvToDF(data_path='./work/user_data/stop_data/test_B.txt', is_test=True)

test_A['label']=res

#copy一份test文件作为最后的保存
test_A_copy=test_A[['query1','query2','label']]
test_A_copy['label1']=-1
#%%In[3] 将文本转变为拼音
from pypinyin import pinyin,Style,lazy_pinyin
test_A['pinyin1']=test_A['query1'].apply(lambda x:pinyin(x, style=Style.TONE3, heteronym=True))
test_A['pinyin2']=test_A['query2'].apply(lambda x:pinyin(x, style=Style.TONE3, heteronym=True))

#%%In[4] 计算文本因为拼音拼错的错别字的编辑距离
import Levenshtein
test_A['dis']=-1

for index,data in enumerate(zip(test_A['pinyin1'],test_A['pinyin2'])):
    distance=0
    if len(data[0])!=len(data[1]):
        distance=-1#不属于错别字范围
    else:
        for i in range(len(data[0])):#遍历query1[[],[]]
            if len(data[0][i])==1:#取一个字的拼音，如果只有一个读音
                if data[0][i][0] not in data[1][i]:#如果该读音不在query2对应字的读音中
                    distance+=1
            else:#如果为多音字
                tmp=0
                for x in data[0][i]:#遍历多音
                    if x not in data[1][i]:#该读音不在query2对应字的读音中
                        tmp+=1
                if tmp==len(data[0][i]):#所有读音都不在query2中
                    distance+=1
    test_A.loc[index,'dis']=distance    

test_A['pinyin1']=test_A['query1'].apply(lazy_pinyin)
test_A['pinyin2']=test_A['query2'].apply(lazy_pinyin)
test_A['Levenshtein2']=list(map(lambda x,y:Levenshtein.distance(x,y),test_A['pinyin1'],test_A['pinyin2']))
pinyin=list(set(test_A[test_A['Levenshtein2']==0].index.tolist()+test_A[test_A['dis']==0].index.tolist()))

#%%In[5] 根据编辑距离特征处理文本
test_A_copy.loc[pinyin,'label1']=1
del test_A['pinyin1'],test_A['pinyin2'],test_A['Levenshtein2'],test_A['dis']
#%%In[8] 插入字
koutou=test_A[test_A_copy['label1']==-1]
koutou=koutou[['query1','query2','label']]
import difflib

#查找同义词或口头语，句子基本上相似，只是替换了个别词
for i,tup in enumerate(zip(koutou.index.tolist(),koutou['query1'],koutou['query2'])):
    str1=tup[1]
    str2=tup[2]
    s = difflib.SequenceMatcher(None, str1, str2)
    word1=[]
    word2=[]
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag!='equal':
            if (i1!=i2):
                word1.append(str1[i1:i2])
            if (j1!=j2):
                word2.append(str2[j1:j2])
    
    koutou.loc[tup[0],'word1']=str(word1)
    koutou.loc[tup[0],'word2']=str(word2)

#%%In[9] word1/word2仅一边插入字
koutou['word1_len']=koutou['word1'].apply(lambda x: 1 if (len(eval(x))==0) else 0) 
koutou_word1_len0=koutou[koutou['word1_len']==1]
del koutou_word1_len0['word1_len']

koutou['word2_len']=koutou['word2'].apply(lambda x: 1 if (len(eval(x))==0) else 0) 
koutou_word2_len0=koutou[koutou['word2_len']==1]
del koutou_word2_len0['word1_len'],koutou_word2_len0['word2_len']

#%%In[10] LAC 预测插入词性
from LAC import LAC
# 装载LAC模型
lac = LAC(mode='lac')
koutou_word1_len0['lac_word']=koutou_word1_len0['word2'].apply(lambda x: lac.run(eval(x)[0]))
koutou_word1_len0['lac_word']=koutou_word1_len0['lac_word'].apply(lambda x:x[1][0])

koutou_word2_len0['lac_word']=koutou_word2_len0['word1'].apply(lambda x: lac.run(eval(x)[0]))
koutou_word2_len0['lac_word']=koutou_word2_len0['lac_word'].apply(lambda x:x[1][0])
#%%In[11] 提取特征 在query2中词性 LOC 
koutou_word1_len0['chaLOC']=list(map(loc_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
#%%In[12] 提取特征 在query2中词性 插入ORG 
koutou_word1_len0['chaORG']=list(map(org_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaORG!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaORG']!=-1]['chaORG']
#%%In[13] 提取特征 在query2中词性 TIME 
koutou_word1_len0['chaTIME1']=list(map(time_0,koutou_word1_len0['word2'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaTIME1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaTIME1']!=-1]['chaTIME1']
#%%In[14] 提取特征 在query2中词性 adj 
koutou_word1_len0['chaADJ1']=list(map(adj_0,koutou_word1_len0['word2'],koutou_word1_len0['query2'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaADJ1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaADJ1']!=-1]['chaADJ1']
#%%In[15] 提取特征 在query2中词性 adj 
koutou_word1_len0['chaADJ2']=list(map(adj_1,koutou_word1_len0['word2'],koutou_word1_len0['query2'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaADJ2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaADJ2']!=-1]['chaADJ2']
#%%In[16] 提取特征 在query2中词性 c 
koutou_word1_len0['chaC']=list(map(c_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaC!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaC']!=-1]['chaC']
#%%In[17] 提取特征 在query2中词性 d 
koutou_word1_len0['chaD']=list(map(d_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaD!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaD']!=-1]['chaD']
#%%In[18] 提取特征 在query2中词性 f 
koutou_word1_len0['chaF']=list(map(f_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaF!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaF']!=-1]['chaF']
#%%In[19] 提取特征 在query2中词性 p 
koutou_word1_len0['chaP']=list(map(p_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaP!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaP']!=-1]['chaP']
#%%In[20] 提取特征 在query2中词性 m
koutou_word1_len0['chaM1']=list(map(m_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaM1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaM1']!=-1]['chaM1']
#%%In[21] 提取特征 在query2中词性 m 
koutou_word1_len0['chaM2']=list(map(m_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaM2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaM2']!=-1]['chaM2']
#%%In[21] 提取特征 在query2中词性 n 
koutou_word1_len0['chaN1']=list(map(n_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaN1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaN1']!=-1]['chaN1']
#%%In[22] 提取特征 在query2中词性 n               
koutou_word1_len0['chaN2']=list(map(n_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaN2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaN2']!=-1]['chaN2']
#%%In[23] 提取特征 在query2中词性 r 
koutou_word1_len0['chaR1']=list(map(r_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaR1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaR1']!=-1]['chaR1']
#%%In[24] 提取特征 在query2中词性 r 
koutou_word1_len0['chaR2']=list(map(r_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaR2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaR2']!=-1]['chaR2']
#%%In[25] 提取特征 在query2中词性 u           
koutou_word1_len0['chaU1']=list(map(u_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaU1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaU1']!=-1]['chaU1']
#%%In 提取特征 在query2中词性 u 
koutou_word1_len0['chaU2']=list(map(u_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaU2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaU2']!=-1]['chaU2']
#%%In[26] 提取特征 在query2中词性 v 
koutou_word1_len0['chaV1']=list(map(v_0,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['query2'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaV1!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaV1']!=-1]['chaV1']
#%%In[27] 在query2中词性 v 
koutou_word1_len0['chaV2']=list(map(v_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaV2!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaV2']!=-1]['chaV2']
#%%In[28] 在query2中词性 xc 
koutou_word1_len0['chaXc']=list(map(xc_1,koutou_word1_len0['word2'],koutou_word1_len0['query1'],koutou_word1_len0['lac_word']))
koutou_word1_len0.loc[koutou_word1_len0.chaXc!=-1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaXc']!=-1]['chaXc']
#%%In[29] 
koutou_word1_len0['chaZ']=list(map(zenm,koutou_word1_len0['word2']))
koutou_word1_len0.loc[koutou_word1_len0.chaZ==1,'chaLOC']=koutou_word1_len0[koutou_word1_len0['chaZ']==1]['chaZ']
#%%In[30]
koutou_word1_len0.loc[koutou_word1_len0.chaLOC!=-1,'label']=koutou_word1_len0[koutou_word1_len0['chaLOC']!=-1]['chaLOC']
#%%In[31]拼接
change=koutou_word1_len0[koutou_word1_len0['label']!=-1]
for tup in zip(change.index.tolist(),change['label']):
     test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[32] 在query1中插入了一些字符
koutou_word2_len0['label1']=list(map(other_1,koutou_word2_len0['word1'],koutou_word2_len0['query1']))
#%%In[33]
koutou_word2_len0['label2']=list(map(other_0,koutou_word2_len0['word1'],koutou_word2_len0['query1'],koutou_word2_len0['query2']))
koutou_word2_len0.loc[koutou_word2_len0.label2!=-1,'label1']=koutou_word2_len0[koutou_word2_len0['label2']!=-1]['label2']
#%%In[34]
koutou_word2_len0.loc[koutou_word2_len0.label1!=-1,'label']=koutou_word2_len0[koutou_word2_len0['label1']!=-1]['label1']
#%%In[35]拼接
change=koutou_word2_len0[koutou_word2_len0['label']!=-1]
for tup in zip(change.index.tolist(),change['label']):
      test_A_copy.loc[tup[0],'label1']=tup[1]

#%%In[35] 查找大学和银行ORG文本
def search_school_bank(x):
    if (('大学' in x[0]) or('银行' in x[0])or ('大学' in x[1]) or('银行' in x[1]))&(('ORG' in x[2])or ('ORG' in x[3])):
        return 1
    else :
        return 0
test_A['school_bank']=list(map(search_school_bank,test_A[['query1','query2','lca_a','lca_b']].values))
school_bank=test_A[(test_A['school_bank']==1)]
school_bank=school_bank[['query1','query2','lca_word1','lca_a','lca_word2','lca_b','label']]
#%%In[36] 对比实体
school_bank['is_eq_org']=list(map(is_eq_org1,school_bank[['query1','query2','lca_word1','lca_a','lca_word2','lca_b']].values))
#%%In[37]
for tup in zip(school_bank.index.tolist(),school_bank['is_eq_org']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[38] 在query1和query2上都有改动的
koutou['word1_len']=koutou['word1'].apply(lambda x: len(eval(x))) 
koutou['word2_len']=koutou['word2'].apply(lambda x: len(eval(x))) 
koutou_word1_word2_len1=koutou[(koutou['word1_len']==1)&(koutou['word2_len']==1)]
#%%In[39] 数字插入
koutou_word1_word2_len1['shuzi1']=koutou_word1_word2_len1['word1'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x))) 
koutou_word1_word2_len1['shuzi2']=koutou_word1_word2_len1['word2'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x)))
#选出全是数字的
koutou_word1_word2_len1['word1_len']=koutou_word1_word2_len1['shuzi1'].apply(lambda x: len(x)) 
koutou_word1_word2_len1['word2_len']=koutou_word1_word2_len1['shuzi2'].apply(lambda x: len(x)) 
shuzi_1=koutou_word1_word2_len1[(koutou_word1_word2_len1['word1_len']==1)&(koutou_word1_word2_len1['word2_len']==0)]
shuzi_2=koutou_word1_word2_len1[(koutou_word1_word2_len1['word1_len']==0)&(koutou_word1_word2_len1['word2_len']==1)]
shuzi=koutou_word1_word2_len1[(koutou_word1_word2_len1['word1_len']!=0)&(koutou_word1_word2_len1['word2_len']!=0)]
#不含数字
wenzi=koutou_word1_word2_len1[(koutou_word1_word2_len1['word1_len']==0)&(koutou_word1_word2_len1['word2_len']==0)]
#%%In[40] shuzi_1(word1是阿拉伯,word2是中文)
shuzi_1['chinese_shuzi']=shuzi_1['shuzi1'].apply(lambda x: convert_to_chinese4(x))
shuzi_1['label1']=list(map(is_shuzi,shuzi_1['chinese_shuzi'],shuzi_1['word2']))
#%%In[41]
shuzi_1['label2']=list(map(lambda x,y: shouji(x,y),shuzi_1['word2'],shuzi_1['query2']))
shuzi_1.loc[shuzi_1.label2!=-1,'label1']=shuzi_1[shuzi_1['label2']!=-1]['label2']
#%%In i和o模糊处理
shuzi_1['label3']=list(map(lambda x,y: mohu(x,y),shuzi_1['word1'],shuzi_1['word2']))
shuzi_1.loc[shuzi_1.label3!=-1,'label1']=shuzi_1[shuzi_1['label3']!=-1]['label3']
#%%In[42] 拼接
for tup in zip(shuzi_1.index.tolist(),shuzi_1['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[43]shuzi_2(word2是阿拉伯,word1是中文)
shuzi_2['chinese_shuzi']=shuzi_2['shuzi2'].apply(lambda x: convert_to_chinese4(x))
shuzi_2['label1']=list(map(is_shuzi,shuzi_2['chinese_shuzi'],shuzi_2['word1']))
#%%In[44]i和o模糊修改
shuzi_2['label2']=list(map(lambda x,y: mohu(x,y),shuzi_2['word1'],shuzi_2['word2']))
shuzi_2.loc[shuzi_2.label2!=-1,'label1']=shuzi_2[shuzi_2['label2']!=-1]['label2']
#%%In[45]拼接
for tup in zip(shuzi_2.index.tolist(),shuzi_2['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[46]shuzi两边均含数字
shuzi['shuzi1']=shuzi['query1'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x))) 
shuzi['shuzi2']=shuzi['query2'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x)))
def is_eq_num(x):#对比两边数字
    for tup in zip(x[7],x[8]):
        if tup[0]!=tup[1]:
            if (('和' in x[0])&('和' in x[1])) | (('与' in x[0])&('与' in x[1])):#并列关系
                if len(list(set(x[7]).difference(set(x[8]))))==0:
                    return 1
            return 0
    if x[3]!=x[4]:
        return 0
    return 1
shuzi['label1']=list(map(lambda x: is_eq_num(x),shuzi.values))
#%%In[47]拼接
for tup in zip(shuzi.index.tolist(),shuzi['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[48] 仅含文字
#将字母转成小写
del wenzi['shuzi1'], wenzi['shuzi2']
wenzi['zimu1']=wenzi['word1'].apply(lambda x:re.findall(r"[a-zA-Z\d]+",str(x))) 
wenzi['zimu2']=wenzi['word2'].apply(lambda x:re.findall(r"[a-zA-Z\d]+",str(x)))

wenzi['word1_len']=wenzi['zimu1'].apply(lambda x: len(x)) 
wenzi['word2_len']=wenzi['zimu2'].apply(lambda x: len(x))
wenzi_replace=wenzi[(wenzi['word1_len']==0)&(wenzi['word2_len']==0)]
#%%  replace的词
wenzi_replace=wenzi_replace[wenzi_replace['word1']!=wenzi_replace['word2']]
#拼接上词性
wenzi_replace['lca_a']=test_A.loc[wenzi_replace.index.tolist(),'lca_a']
wenzi_replace['lca_word1']=test_A.loc[wenzi_replace.index.tolist(),'lca_word1']
wenzi_replace['lca_b']=test_A.loc[wenzi_replace.index.tolist(),'lca_b']
wenzi_replace['lca_word2']=test_A.loc[wenzi_replace.index.tolist(),'lca_word2']
#去除school_bank
school_bank_list=school_bank.index.tolist()
wenzi_replace_list=wenzi_replace.index.tolist()
exschoolbank_list=list(set(wenzi_replace_list).difference(set(school_bank_list)))
wenzi_replace_exschoolbank=wenzi_replace[wenzi_replace.index.isin(exschoolbank_list)]
#%%In[49] 查找PER
del wenzi_replace_exschoolbank['word1_len'] ,wenzi_replace_exschoolbank['word2_len'],wenzi_replace_exschoolbank['zimu1'],wenzi_replace_exschoolbank['zimu2']
#寻找per位置上词
wenzi_replace_exschoolbank['per_1']=wenzi_replace_exschoolbank['lca_a'].apply(lambda x:1 if 'PER' in eval(x) else 0)
wenzi_replace_exschoolbank['per_2']=wenzi_replace_exschoolbank['lca_b'].apply(lambda x:1 if 'PER' in eval(x) else 0)
#PER
wenzi_replace_PER=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['per_1']==1)&(wenzi_replace_exschoolbank['per_2']==1)]
wenzi_replace_PER1=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['per_1']==1)&(wenzi_replace_exschoolbank['per_2']==0)]
wenzi_replace_PER2=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['per_1']==0)&(wenzi_replace_exschoolbank['per_2']==1)]
#%%In[50] PER
wenzi_replace_PER['label1']=list(map(is_eq_per,wenzi_replace_PER.values))
#%%In[51] PER 
change=wenzi_replace_PER[wenzi_replace_PER['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In PER
wenzi_replace_PER1['label1']=list(map(is_eq_per2,wenzi_replace_PER1.values))
#%% PER 
change=wenzi_replace_PER1[wenzi_replace_PER1['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[52] 查找除了大学和银行ORG
del wenzi_replace_exschoolbank['per_1'],wenzi_replace_exschoolbank['per_2']
wenzi_replace_exschoolbank['org_1']=wenzi_replace_exschoolbank['lca_a'].apply(lambda x:1 if 'ORG' in eval(x) else 0)
wenzi_replace_exschoolbank['org_2']=wenzi_replace_exschoolbank['lca_b'].apply(lambda x:1 if 'ORG' in eval(x) else 0)

wenzi_replace_ORG=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['org_1']==1)&(wenzi_replace_exschoolbank['org_2']==1)]
#%%In[53] ORG
wenzi_replace_ORG['label1']=list(map(is_eq_org2,wenzi_replace_ORG.values))
#%%In[54] ORG 
change=wenzi_replace_ORG[wenzi_replace_ORG['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[55] 查找LOC
del wenzi_replace_exschoolbank['org_1'],wenzi_replace_exschoolbank['org_2']
wenzi_replace_exschoolbank['loc_1']=wenzi_replace_exschoolbank['lca_a'].apply(lambda x:1 if 'LOC' in eval(x) else 0)
wenzi_replace_exschoolbank['loc_2']=wenzi_replace_exschoolbank['lca_b'].apply(lambda x:1 if 'LOC' in eval(x) else 0)

wenzi_replace_LOC=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['loc_1']==1)&(wenzi_replace_exschoolbank['loc_2']==1)]
#%%In[56] LOC
wenzi_replace_LOC['label1']=list(map(is_eq_loc,wenzi_replace_LOC.values))
#%%In[57] LOC 
change=wenzi_replace_LOC[wenzi_replace_LOC['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[58] 查找TIME
del wenzi_replace_exschoolbank['loc_1'],wenzi_replace_exschoolbank['loc_2']
wenzi_replace_exschoolbank['time_1']=wenzi_replace_exschoolbank['lca_a'].apply(lambda x:1 if 'TIME' in eval(x) else 0)
wenzi_replace_exschoolbank['time_2']=wenzi_replace_exschoolbank['lca_b'].apply(lambda x:1 if 'TIME' in eval(x) else 0)

wenzi_replace_time=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['time_1']==1)&(wenzi_replace_exschoolbank['time_2']==1)]
wenzi_replace_time1=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['time_1']==1)&(wenzi_replace_exschoolbank['time_2']==0)]
wenzi_replace_time2=wenzi_replace_exschoolbank[(wenzi_replace_exschoolbank['time_1']==0)&(wenzi_replace_exschoolbank['time_2']==1)]
#%%In[59] TIME1
wenzi_replace_time1['label1']=list(map(time1,wenzi_replace_time1['word1'],wenzi_replace_time1['word2']))
#%%In[60] TIME1
change=wenzi_replace_time1[wenzi_replace_time1['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[61] TIME2
wenzi_replace_time2['label1']=list(map(time2,wenzi_replace_time2['word1'],wenzi_replace_time2['word2']))
#%%In[62] TIME2 
change=wenzi_replace_time2[wenzi_replace_time2['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%In[63] TIME
wenzi_replace_time['label1']=list(map(time_,wenzi_replace_time.values))
#%%In[64] TIME
change=wenzi_replace_time[wenzi_replace_time['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label1']=tup[1]
#%%
change=test_A_copy[test_A_copy['label1']!=-1]
for tup in zip(change.index.tolist(),change['label1']):
      test_A_copy.loc[tup[0],'label']=tup[1]
test_A_copy['label']=test_A_copy['label'].astype(int)
#%%In[65] 最后保存结果
test_A_copy['label'].to_csv('./work/user_data/tmp_result/cyw.csv',header=None, index=False)

import pandas as pd
import numpy as np

test=pd.read_csv('./work/raw_data/test_B_1118.tsv',sep='\t',names=['text_one','text_two'])
label=pd.read_csv('./work/user_data/tmp_result/cyw.csv',names=['label'])
test['label']=label['label']
#print("处理前正例样本数：",len(test[test['label']==1]))
import json
def load_dict(filename):
    '''load dict from json file'''
    '''读取json文件'''
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic

def insert_list(text1,text2):
    '''

    得到两个文本中不同的部分
    ----------
    text1 : 文本1
        测试集文本1
    text2 : 文本2
        测试集文本2

    Returns：得到插入的文本

    '''
    tmp=1
    data=[]
    for i in text1:
        if i not in text2:
            tmp=0
    if tmp==1:
        for i in text2:
            if i not in text1:
                data.append(i)
    return ''.join(data)



test['new_insert_one_list']=list(map(lambda x,y:insert_list(x,y),test['text_one'],test['text_two']))
test['new_insert_two_list']=list(map(lambda x,y:insert_list(y,x),test['text_one'],test['text_two']))

#%%
def is_in_family(text1,text2,texta,textb):
    '''
    文本插入部分中的内容是否属于家人
    ----------
    text1 : 文本1和文本2不同的部分
    text2 : 文本2和文本1不同的部分
    texta : 文本1
    textb : 文本2
    Returns
    '''
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    
    things=dic['family_list']
    nothings=dic['family_same_list']
    for i in things:
        if i in text1:
            for j in nothings:
                if j in texta and j in textb:
                    return 2
            return 1
    for i in things:
        if i in text2:
            for j in nothings:
                if j in texta and j in textb:
                    return 2
            return 1
    return 0
test['is_family']=list(map(lambda x,y,z,r:is_in_family(x,y,z,r),test['new_insert_one_list'],test['new_insert_two_list'],test['text_one'],test['text_two']))
#%%
def is_in_ORG(text1,text2):
    '''
    获取文本中的实体信息是否是银行
    ----------
    text1 : 文本1
    text2 : 文本2
    Returns
    '''
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    
    things=dic['bank_things']
    tmp=0
    for i in things:
        if i in text1:
            thing=dic['bank_things']
            thing.remove(i)
            for j in thing:
                if j in text2:
                    tmp=1
    return tmp
test['is_ORG']=list(map(lambda x,y:1 if is_in_ORG(x,y) or is_in_ORG(y,x) else 0,test['text_one'],test['text_two']))
#%%
import re
'''
正则匹配文本中的数字
'''
test['insert_one_num']=test['text_one'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x)))
test['insert_two_num']=test['text_two'].apply(lambda x:re.findall(r"\d+\.?\d*",str(x)))

test['no_equal_num']=list(map(lambda x,y:1 if x!=y and x!=[] and y!=[] and len(x)==1 and len(y)==1 else 0,test['insert_one_num'],test['insert_two_num']))
test['one_num']=test['text_one'].apply(lambda x:1 if re.findall(r"\d+\.+[\u4e00-\u9fa5]",str(x)) or re.findall(r"\d+[一二三四五六七八九十]+[险]",str(x)) or re.findall(r"\d\.\ ",str(x))  or re.findall(r"\d+[克]",str(x)) or re.findall(r"100000",str(x)) else 0 )
test['two_num']=test['text_two'].apply(lambda x:1 if re.findall(r"\d+\.+[\u4e00-\u9fa5]",str(x)) or re.findall(r"\d+[一二三四五六七八九十]+[险]",str(x)) or re.findall(r"\d\.\ ",str(x)) or re.findall(r"1+[kg]",str(x)) or re.findall(r"10万",str(x))else 0 )
test['onetwo_num']=list(map(lambda x,y:1 if x==1 and y==1 else 0,test['one_num'],test['two_num']))


tmp2=test.copy()  
    
#%%
test.loc[test.is_family==1,'label']=0
test.loc[test.is_family==2,'label']=1
test.loc[test.is_ORG==1,'label']=0
test.loc[test.no_equal_num==1,'label']=0
test.loc[test.onetwo_num==1,'label']=1
#print("处理完num-family-bank后：",len(test[test['label']==1]))
test['label'].to_csv('./work/user_data/tmp_result/deal1.csv',header=None,index=False)


#%%

import pandas as pd
import numpy as np
from pypinyin import lazy_pinyin
import numpy as np
from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')
#test=pd.read_csv('E:/Competition/Compare_text/result/test_B/test_B_1118.tsv',sep='\t',names=['text_one','text_two'])
#label=pd.read_csv('E:/Competition/Compare_text/result/test_B/deal_wll1_cyw.csv',names=['label'])
#test['label']=label['label']
test=test[['text_one','text_two','label']]
#利用百度的LAC获得文本的分词和词性
test['lac_one']=test['text_one'].apply(lambda x:lac.run(x)[0])
test['lac_two']=test['text_two'].apply(lambda x:lac.run(x)[0])

test['seg_one']=test['text_one'].apply(lambda x:lac.run(x)[1])
test['seg_two']=test['text_two'].apply(lambda x:lac.run(x)[1])

#print("处理similar前：",len(test[test['label']==1]))
#%%
def lac_similiar(text1,text2):
    '''
    比较分词后的text1和text2列表是否一样
    ----------
    text1 : 分词后的text1
    text2 : 分词后的text2
    '''
    tmp=1
    for i in text1:
        if i not in text2:
            tmp=0
    for i in text2:
        if i not in text1:
            tmp=0
    return tmp
test['similar']=list(map(lambda x,y,z,r:1 if lac_similiar(x, y) and len(z)==len(r) else 0,test['lac_one'],test['lac_two'],test['text_one'],test['text_two']))
#%%
import re
def mean(text1):
    '''
    词性为LOC的文本匹配meanthings得到文本
    ----------
    text1 :文本1
    '''
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    
    mean_things=dic['LOC_mean_things']
    for i in mean_things:
        if i in text1 and '隔离' not in text1:
            return 1
    return 0

def location(text):
    '''
    词性为LOC的文本匹配nomean_things得到文本
    ----------
    text1 :文本1
    '''
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    nomean_things=dic['LOC_nomean_things']
    for i in nomean_things:
        if i in text or ('到' in text and '多远' not in text and '多少公里' not in text):
            return 1
    return 0

def re_seg_mean(text1,seg1):
    '''
    根据词性匹配相应文本
    ----------
    text1 : 文本1
    seg1 : 文本词性
    -------
    int
        DESCRIPTION.

    '''
    line=''.join(seg1)
    pattern1=re.compile('nvvnxc|PERvrnwnvrxc')
    r1=pattern1.search(line)
    if r1:
        seg10=['喂','浇','有']
        for i in seg10:
            if i in text1:
                return 1
    return 0

def is_nvn__(text1,seg1):
    '''
    根据词性匹配相应文本
    ----------
    text1 : 文本1
    seg1 : 文本词性
    '''
    #lac分词有名x词类型的词语
    #将词性为nx类型的都转化为n
    dic1=load_dict('./work/user_data/stop_data/stopword_2')
    seg=[]
    for i in seg1:
        if i[0]=='n':
            seg.append('n')
        else:
            seg.append(i)
    #根据词性匹配简单主谓宾形式的文本
    line=''.join(seg)
    pattern1=re.compile('nvnxc|PERvnxc')
    a=pattern1.search(line)
    
    if a and 'vn' not in seg1 and 'ad' not in seg1 and 'p' not in seg1 and 'k7' not in text1:
        if 'PER' in seg:
            if seg.count('PER')==1:
                return 1
            else:
                return 0
        else:
            return 1
        
    #根据词性匹配特别谓语动词的文本
    line=''.join(seg1)
    pattern2=re.compile('nvn|PERvn|nvvn|auvn|nzvnz')
    b=pattern2.search(line)
    if b and 'nn' not in line:
        dic=dic1['nvn_pattern2']
        for i in dic:
            if i in text1:
                  return 1 
    
    #根据词性匹配negative情感的文本
    line=''.join(seg1)
    pattern3=re.compile('dvdv|nvnv|dvnd|vvrn|nrvm|nzrvn')
    c=pattern3.search(line)
    if c:
        if re.findall(r"只+[\u4E00-\u9FA5]+不|[\u4E00-\u9FA5]+未|有点+[\u4E00-\u9FA5]+不|怕+[\u4E00-\u9FA5]|[\u4E00-\u9FA5]+次|怎么+[\u4E00-\u9FA5]+入",text1):
            return 1
    
    #根据词性匹配包含方向介词的文本
    line=''.join(seg1)
    pattern3=re.compile('npnrvn|npnfvrv|ORGnnrvm|npvaxc|npnaxc|anpnapn|nzpnz|npnun|ncnun')
    d=pattern3.search(line)
    if d and re.findall(r"比+[\u4E00-\u9FA5]|往+[\u4E00-\u9FA5]+长|[\u4E00-\u9FA5]+敷|在+[\u4E00-\u9FA5]+在|[\u4E00-\u9FA5]+对+[\u4E00-\u9FA5]",text1):
        return 1
    
    #根据词性匹配包含数字、包含定语的文本
    line=''.join(seg1)
    pattern4=re.compile('mvmvn|mnzvv|vncnrvn|vnnun|nvnunv|nvncn|pnun|nnun|fnfn')
    d=pattern4.search(line)
    if d:
        if re.findall(r"[一二三四五六七八九十]+[\u4E00-\u9FA5]+[一两三四五六七八九十]|和+[\u4E00-\u9FA5]+算|写+[\u4E00-\u9FA5]+的|还是+[\u4E00-\u9FA5]\
                      |上+[\u4E00-\u9FA5]+下|带+[\u4E00-\u9FA5]+的|历+[\u4E00-\u9FA5]+还是|于+[\u4E00-\u9FA5]+的|手+[\u4E00-\u9FA5]+的",text1):
            return 1
    
    return 0
    
#和上文相同得到插入进文本1文本2的文本
test['insert_one_list']=list(map(lambda x,y:insert_list(x,y),test['text_one'],test['text_two']))
test['insert_two_list']=list(map(lambda x,y:insert_list(y,x),test['text_one'],test['text_two']))

#词性为LOC的匹配
test['is_mean']=list(map(lambda x,y,z:1 if ('LOC' in y or '-1' in x or '20' in x) and mean(x) and z==1 else 0,test['text_one'],test['seg_one'],test['similar']))#这些都是1

#词性为LOC的匹配
test['is_loc']=list(map(lambda x,y,z:1 if '多少公里' not in x and '多远' not in x and location(x) and 'LOC' in y and z==1 else 0,test['text_one'],test['seg_one'],test['similar']))#这些都是0

#词性为PER或LOC实体匹配
test['lac_LOC_PER']=list(map(lambda x,y,z,r:1 if ('LOC' in x and 'LOC' in y or 'PER' in x and 'PER' in y and 'v' not in x)and (r=='小')  else 0,test['seg_one'],test['seg_two'],test['insert_one_list'],test['insert_two_list']))

#句子词性主谓宾关系匹配
test['is_nvn__']=list(map(lambda x,y,z:1 if is_nvn__(x,y) and z==1 else 0,test['text_one'],test['seg_one'],test['similar']))
test['is_nvn_mean']=list(map(lambda x,y,z:1 if re_seg_mean(x,y) and z==1 else 0,test['text_one'],test['seg_one'],test['similar']))

#比较关系
test['is_geng']=list(map(lambda x,y:1 if ('比' in x or '比' in y)and(('更多' in x or'更厉害' in x or'更漂亮' in x or'更高'in x or '更可爱'in x) or ('更多' in y or'更厉害' in y or'更漂亮' in y or'更高'in y or '更可爱'in y))and 'cad' not in x 
                         else 0,test['text_one'],test['text_two']))

#时间关系
test['is_yiciduojiu']=list(map(lambda x,y:1 if'一次多久' in x and y.index('多久')<y.index('一次') or '多久一次' in x and y.index('多久')>y.index('一次') else 0,
                               test['text_one'],test['lac_two']))

#在文本长度和元素组成都相同的情况下匹配数字
test['one_num']=list(map(lambda x,y:re.findall(r"\d+\.?\d*",str(x)) if y==1 and '×' in x  else 0,test['text_one'],test['similar']))


#表达并列关系的连词
test['one_lian']=list(map(lambda x,y:1 if re.findall(r"没+[\u4E00-\u9FA5]+也|没+[\u4E00-\u9FA5]+又",str(x)) and y==1 else 0,test['text_one'],test['similar']))


'''xx1=test[test['is_mean']==1]
xx2=test[test['is_loc']==1]
xx3=test[test['is_nvn__']==1]
xx3_1=test[test['is_nvn_mean']==1]
xx4=test[test['is_geng']==1]
xx5=test[test['is_yiciduojiu']==1]
xx6=test[test['one_num']!=0]
xx7=test[test['one_lian']==1]'''
#%%
test.loc[test.is_mean==1,'label']=1
test.loc[test.is_loc==1,'label']=0
test.loc[test.lac_LOC_PER==1,'label']=0
test.loc[test.is_nvn__==1,'label']=0
test.loc[test.is_nvn_mean==1,'label']=1
test.loc[test.is_geng==1,'label']=1
test.loc[test.is_yiciduojiu==1,'label']=0
test.loc[test.one_num!=0,'label']=1
test.loc[test.one_lian==1,'label']=1
#print("处理similar后：",len(test[test['label']==1]))
tmp1=test[test['similar']==1]
#%%
test['label'].to_csv('./work/user_data/tmp_result/deal2.csv',header=None,sep='\t',index=None)

#%%

import pandas as pd
import numpy as np
from pypinyin import lazy_pinyin
import numpy as np
from LAC import LAC


# 装载LAC模型
#lac = LAC(mode='lac')
#test=pd.read_csv('E:/Competition/Compare_text/result/test_B/test_B_1118.tsv',sep='\t',names=['text_one','text_two'])
#label=pd.read_csv('E:/Competition/Compare_text/result/test_B/deal_wll1+2_cyw.csv',names=['label'])
#test['label']=label['label']
#test['lac_one']=test['text_one'].apply(lambda x:lac.run(x)[0])
#test['lac_two']=test['text_two'].apply(lambda x:lac.run(x)[0])

#test['seg_one']=test['text_one'].apply(lambda x:lac.run(x)[1])
#test['seg_two']=test['text_two'].apply(lambda x:lac.run(x)[1])

test=test[['text_one','text_two','label','lac_one','lac_two','seg_one','seg_two']]
#print("处理most_similar前：",len(test[test['label']==1]))
#%%
def most_similiar(text1,text2):
    '''
    

    Parameters
    ----------
    text1 : 分词后的文本1
    text2 : 分词后的文本2
    Returns
    -------
    data : 分词后两个文本的差异词汇

    '''
    tmp=0
    data=[]
    for i in text1:
        if i not in text2:
            tmp=tmp+1
            data.append(i)
    for i in text2:
        if i not in text1:
            tmp=tmp+1
            data.append(i)
    return data

test['most_similar']=list(map(lambda x,y,z,r:1 if len(most_similiar(x, y))<=2 and len(most_similiar(x, y))>0 and len(z)==len(r) else 0,test['lac_one'],test['lac_two'],test['text_one'],test['text_two']))
test['not_similar_list']=list(map(lambda x,y,z,r:most_similiar(x, y) if len(most_similiar(x, y))<=2 and len(most_similiar(x, y))>0 and len(z)==len(r) else 0,test['lac_one'],test['lac_two'],test['text_one'],test['text_two']))

#%%
def not_similat_list_seg(x,y,z,i,j):
    '''
    Parameters
    ----------
    x : text1分词后的词性
    y : text2分词后的词性
    z : 文本长度是否相同
    i : text1分词列表
    j : text1分词列表
    return：文本长度相同的样本不同词汇的词性
    '''
    seg=[]
    if z==0:
        return 0
    for a in z:
        if a in i:
            b=i.index(a)
            seg.append(x[b])
        if a in j:
            b=j.index(a)
            seg.append(y[b])
    return seg

import re
 
def check(str):
    '''
    Parameters
    ----------
    str : 字符串

    Returns：检查字符串是否都为英文字符
    -------
    int
        DESCRIPTION.

    '''
    my_re = re.compile(r'[A-Za-z]',re.S)
    res = re.findall(my_re,str)
    if len(res):
        return 1
    else:
        return 0

def MM(x):
    '''

    Parameters
    ----------
    x : 词性都为数量词的文本不相同词语
    Returns：根据量词集合返回文本是否量词不同
    -------
    tmp : TYPE
        DESCRIPTION.

    '''
    x1=x[0]
    x2=x[1]
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    things=dic['mm_things']
    tmp=0
    for i in things:
        if i in x1:
            thing=dic['mm_things']
            thing.remove(i)
            for j in thing:
                if j in x2:
                    tmp=1
    if '一颗' in x1 and '一次' in x2:
        tmp=1
    elif '一项' in x1 and '一点' in x2:
        tmp=1
    elif '一颗' in x1 and '一套' in x2:
        tmp=1
    elif '一回' in x1 and '一集' in x2:
        tmp=1
    return tmp

def antonym(x):
    '''

    Parameters
    ----------
    x : 列表大小为2的不同词汇列表
    Returns：根据反义词或表达不同的词汇集合得出文本是否为反义
    -------

    '''
    
    dic=load_dict('./work/user_data/stop_data/stopword_2')
    x1=x[0]
    x2=x[1]
    left=dic['antonym_left']
    right=dic['antonym_right']
    for i in left:
        if i in x1:
            num=left.index(i)
            if right[num] in x2:
                return 1
    return 0

def nandao(x,y):
    '''
    Parameters
    ----------
    x : 文本1
    y : 文本2
    Returns：文本是否为反问句

    '''
    if '难道' in x and '不'  in x:
        x.strip('难道')
        x.strip('不')
        x.strip('吗')
        if '吗' not in y and '不' not in y and '难道' not in y and (x==y or y in x or len(x.strip(y))<=2 or len(y.strip(x))<=2):
            return 1
    return 0



#得到文本1和文本2不同词汇的词性列表
test['not_similar_list_seg']=list(map(lambda x,y,z,i,j:not_similat_list_seg(x, y, z, i, j),test['seg_one'],test['seg_two'],test['not_similar_list'],test['lac_one'],test['lac_two']))
#得到文本1和文本2不同词汇的词性列表的列表长度为2的文本
#即两个文本大致相同只有一个词有差异
test['is_equal_most_similar']=test['not_similar_list_seg'].apply(lambda x:1 if x!=0 and len(x)==2 and x[0]==x[1] else 0)


#两个文本中有差异的词的词性都为LOC的文本
test['is_LOC']=test['not_similar_list_seg'].apply(lambda x:1 if x==['LOC','LOC'] else 0)
#两个文本中有差异的词的词性都为数量的文本
test['is_m']=list(map(lambda x,y:1 if x==['m','m'] and MM(y) else 0,test['not_similar_list_seg'],test['not_similar_list']))
#两个文本中有差异的词互为反义词或差异较大的词
test['is_antonym']=test['not_similar_list'].apply(lambda x:1 if x!=0 and len(x)==2 and antonym(x) else 0)
#两个文本是反问句和陈述句的关系
test['is_nandao']=list(map(lambda x,y:nandao(x,y),test['text_one'],test['text_two']))



'''x=test[test['is_LOC']==1]
x2=test[test['is_m']==1]
x3=test[test['is_antonym']==1]
x4=test[test['is_nandao']==1]'''

#%%
test.loc[test.is_LOC==1,'label']=0
test.loc[test.is_m==1,'label']=0
test.loc[test.is_antonym==1,'label']=0
test.loc[test.is_nandao==1,'label']=1
#%%
#print("处理most_similar_wll_all后：",len(test[test['label']==1]))
test['label'].to_csv('./work/prediction_result/final_result.csv',header=None,sep='\t',index=None)

