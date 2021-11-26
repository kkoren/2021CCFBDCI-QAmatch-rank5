# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christina
"""

'''
工具类
用于对文本进行修改过滤，提取文本特征，修正LAC分词错误
'''


#加载字典
import json
def load_dict(filename):
 '''load dict from json file'''
 with open(filename,"r") as json_file:
  dic = json.load(json_file)
 return dic


def convert_to_chinese4(x):
    '''
    将数字转为阿拉伯数字
    '''
    if '.' in x[0]:
        return x[0]
    if '00' ==x[0]:
        return '点'
    num=int(x[0])
    _MAPPING = (u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'十一', u'十二', u'十三', u'十四', u'十五', u'十六', u'十七',u'十八', u'十九')

    _P0 = (u'', u'十', u'百', u'千',)

    _S4 = 10 ** 4

    if (0 <= num and num < _S4):
        if num < 20:
            return _MAPPING[num]
        else:
            lst = []
            while num >= 10:
                lst.append(num % 10)
                num = num / 10
            lst.append(num)
            c = len(lst)  # 位数
            result = u''
            for idx, val in enumerate(lst):
                val = int(val)
                if val != 0:
                    result += _P0[idx] + _MAPPING[val]
                    if idx < c - 1 and lst[idx + 1] == 0:
                        result += u'零'
            if result[::-1]=='一百':
                return '百'
            if result[::-1]=='二百五十':
                return '笨'
            return result[::-1]
    else:
         return str(num)

#根据LAC所拥有的词性分类
#判断是否插入LOC
def loc_0(word2,query1,lac_word):
    '''
    判断是否插入LOC
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    loc_dict_0=dic['loc_dict_0']
    if 'LOC' ==lac_word:#插入词性是LOC
        if eval(word2)[0] in loc_dict_0:
            li=loc_dict_0.get(eval(word2)[0])
            for s in li:
                if (s in query1) or (eval(word2)[0] in query1):
                    return 1
            return 0
    return -1

#判断是否插入ORG
def org_0(word2,query1,lac_word):
    '''
    判断是否插入ORG
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    org_dict_0=dic['org_dict_0']
    if 'ORG'==lac_word:
        if eval(word2)[0] in org_dict_0:
            li=org_dict_0.get(eval(word2)[0])
            for s in li:
                if (s in query1) or (eval(word2)[0] in query1):
                    return 1
            return 0
    return -1

#判断是否插入词性TIME
def time_0(word2,lac_word):
    '''
    判断是否插入词性TIME
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    time_dict_0=dic['time_dict_0']
    if 'TIME'==lac_word:
        for time in list(time_dict_0.keys()):
            for word in eval(word2):
                if (time in word):
                    return 0
    return -1

#判断是否插入词性adj
#颜色形容词
color_adj=['红','橙','黄','绿','青','黑']
stopword=['新鲜','干净']+['生涯','生理','一生','先生']#停用词
def adj_0(word2,query2,lac_word):
    '''
    判断是否插入词性adj
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    adj_dict_0=dic['adj_dict_0']
    if ('a'==lac_word)or('an'==lac_word)or('ad'==lac_word):#词性为adj
        for adj in list(adj_dict_0.keys()):#遍历字典
            for word in eval(word2):#遍历分词
                if (adj=='快')&(adj in word):
                    for tmp in adj_dict_0[adj]:
                        if tmp in query2:
                            return 0
                    return -1
                elif (adj=='酸')&(adj in word):
                    if word==adj:
                        for tmp in adj_dict_0[adj]:
                            if tmp in query2:
                                return 1
                        return 0
                    else:
                        return -1
                elif(adj=='甜')&(word==adj):
                    for tmp in adj_dict_0[adj]:
                        if (tmp not in query2):
                            return 0
                        else:
                            return -1
                elif (adj=='老')&(word==adj):
                    for tmp in adj_dict_0[adj]:
                        if tmp in query2:
                            return 1
                    return 0
                elif (adj in color_adj)&(word==adj):
                    for tmp in adj_dict_0[adj]:
                        if tmp in query2:
                            return 0
                elif (adj in word)&(word not in stopword):
                    return 0
    return -1
 
#判断是否插入词性adj
def adj_1(word2,query2,lac_word):
    '''
    判断是否插入词性adj
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    adj_dict_1=dic['adj_dict_1']
    if ('a'==lac_word)or('ad'==lac_word):
        for adj in list(adj_dict_1.keys()):
            for word in eval(word2):
                if adj in word:
                    return 1
    return -1

#判断是否插入词性c
def c_0(word2,query1,lac_word):
    '''
    判断是否插入词性c
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    c_dict_0=dic['c_dict_0']
    if 'c'==lac_word:
        if eval(word2)[0] in c_dict_0:
            li=c_dict_0.get(eval(word2)[0])
            for s in li:
                if (s in query1) or (eval(word2)[0] in query1):
                    return 1
            return 0
    return -1

#判断是否插入词性d
def d_0(word2,query1,lac_word):
    '''
    判断是否插入词性d
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    d_dict_0=dic['d_dict_0']
    if 'd'==lac_word:
        for d in d_dict_0:
            for word in eval(word2):
                if (d=='不'):
                    if (word=='不')&('难道' not in word2):
                        return 0
                elif ( d in word):
                    return 0
    return -1

#判断是否插入词性f
def f_0(word2,query1,lac_word):
    '''
    判断是否插入词性f
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    f_dict_0=dic['f_dict_0']
    if ('f'==lac_word)or('m'==lac_word):
        for f in f_dict_0:
            for word in eval(word2):
                if ( f in word):
                    for tmp in f_dict_0[f]:
                        if tmp in query1:
                            return 1
                    return 0
    return -1

#判断是否插入词性p
def p_1(word2,query1,lac_word):
    '''
    判断是否插入词性p
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    p_dict_1=dic['p_dict_1']
    if ('p'==lac_word):
        for d in p_dict_1:
            for word in eval(word2):
                if ( d in word):
                    return 1
    return -1

#判断是否插入词性m
def m_0(word2,query1,lac_word):
    '''
    判断是否插入词性m
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    m_dict_0=dic['m_dict_0']
    if ('m'==lac_word):
        for m in m_dict_0:
            for word in eval(word2):
                if word==m:
                    return 0
    return -1

#判断是否插入词性m
def m_1(word2,query1,lac_word):
    '''
    判断是否插入词性m
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    m_dict_1=dic['m_dict_1']
    if ('m'==lac_word):
        for m in m_dict_1:
            for word in eval(word2):
                if (len(eval(word2))==1)&(m==word):
                    for tmp in m_dict_1[m] :
                        if tmp in query1:
                            return -1
                    return 1
    return -1

#判断是否插入词性n
def n_0(word2,query1,lac_word):
    '''
    判断是否插入词性n
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    ''' 
    dic=load_dict('./work/user_data/stop_data/stopword')
    n_dict_0=dic['n_dict_0']
    n_school=dic['n_school']
    n_dict=dict(n_dict_0, **n_school)
    if ('n'==lac_word)or('nz' ==lac_word):
        for n in n_dict:
            for word in eval(word2):
                if n in word:
                    if word in stopword:
                        return -1
                    if word in dic['n_adj']:
                        for tmp in n_dict_0[n]:
                            if tmp in query1:
                                return 0
                        return -1
                    if n in n_school:
                        for tmp in n_school[n]:
                            if tmp in query1:
                                return 1
                    if n in dic['n_li']:
                        for tmp in n_dict_0[n]:
                            if tmp in query1:
                                return -1
                        return 0
                    return 0
    return -1

#判断是否插入词性n
def n_1(word2,query1,lac_word):
    '''
    判断是否插入词性n
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    n_dict_1=dic['n_dict_1']
    if ('n'==lac_word):
        for n in n_dict_1:
            for word in eval(word2):
                if n in word:
                    if n in dic['n_n']:
                        for tmp in n_dict_1[n]:
                            if (tmp in query1)or(tmp in word2):
                                return 0
                            else:
                                return 1
                    if n=='吗':
                        for tmp in n_dict_1[n]:
                            if tmp in query1:
                                return -1
                            else:
                                return 1

                    return 1
    return -1
#判断是否插入词性r
def r_0(word2,query1,lac_word):
    '''
    判断是否插入词性r
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    r_dict_0=dic['r_dict_0']
    if ('r'==lac_word)or('w'==lac_word):
        for n in r_dict_0:
            for word in eval(word2):
                if n in word:
                   return 0
    return -1

#判断是否插入词性r
def r_1(word2,query1,lac_word):
    '''
    判断是否插入词性r
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    r_dict_1=dic['r_dict_1'] 
    if ('r'==lac_word):
        for n in r_dict_1:
            for word in eval(word2):
                if n in word:
                    if (n=='你'):
                        if (n==word)&('可以' not in query1)&('现在' not in word2)&('为什么' not in word2):
                            return 1
                        else:
                            return -1
                    if n=='这个' :
                        if (n==word)& (len(eval(word2))==1):
                            return 1
                        else:
                            return -1
                    return 1
    return -1
#判断是否插入词性u
def u_0(word2,query1,lac_word):
    '''
    判断是否插入词性u
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    u_dict_0=dic['u_dict_0']
    if ('u'==lac_word):
        for n in u_dict_0:
            for word in eval(word2):
                if n in word:
                    for tmp in u_dict_0[n]:
                        if tmp in query1:
                            return -1
                    return 0
    return -1
#判断是否插入词性u
def u_1(word2,query1,lac_word):
    '''
    判断是否插入词性u
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    u_dict_1=dic['u_dict_1']
    if ('u'==lac_word):
        for n in u_dict_1:
            for word in eval(word2):
                if n in word:
                    return 1
    return -1

#判断是否插入词性v 
def v_0(word2,query1,query2,lac_word):
    '''
    判断是否插入词性v
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    v_dict_0=dic['v_dict_0']
    if ('v'==lac_word):
        for v in v_dict_0:
            for word in eval(word2):
                if (word=='没有')&(query2[-2:]!=word):
                    return 0
                if v in word:
                    if v in dic['v_li']:
                        if (v==word):
                            return 0
                        else:
                            return -1
                    return 0
    return -1

#判断是否插入词性v
def v_1(word2,query1,lac_word):
    '''
    判断是否插入词性v
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 判断是否插入词性v
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    v_dict_1=dic['v_dict_1']
    if ('v'==lac_word)or ('vn'==lac_word)or ('vd'==lac_word):
        for n in v_dict_1:
            for word in eval(word2):
                if n in word:
                    if n in dic['v_v']:
                        if (len(eval(word2))==1)&('用英语' not in query1):
                            return 1
                        else:
                            return -1
                    if n=='吗':
                        for tmp in v_dict_1[n]:
                            if tmp in word2:
                                return 0
                        return 1
                    return 1
    return -1
    return -1
#判断是否插入词性xc
def xc_1(word2,query1,lac_word):
    '''
    判断是否插入词性xc
    Parameters
    ----------
    word2 : str
        query2 按词性分词.
    query1 : str
        query1文本.
    lac_word : str
        word2 词性.

    Returns 判断是否插入词性xc
    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    xc_dict_1=dic['xc_dict_1']
    if ('xc'==lac_word):
        for n in xc_dict_1:
            for word in eval(word2):
                if n in word:
                    if (n=='吗'):#门句判断
                        for tmp in xc_dict_1[n]:
                            if tmp in query1:
                                return 1
                        return -1
                    return 1
    return -1

def other_1(word1,query1):
    dic=load_dict('./work/user_data/stop_data/stopword')
    other_dict1=dic['other_dict1']
    for word in eval(word1):
        if word=='.00':
            return 1
        for adv in other_dict1:
            if adv in dic['adv']:
                if (adv==word):
                    for tmp in other_dict1[adv]:
                        if tmp in query1:
                            return -1
                    return 1
            elif adv in word:
                for tmp in dic['li2']:
                    if tmp in query1:
                        return -1
                return 1
    return -1

def other_0(word1,query1 ,query2):
    dic=load_dict('./work/user_data/stop_data/stopword')
    other_dict0=dic['other_dict0']
    if query2=='是什么意思':
        return 0
    for word in eval(word1):
        for adv in other_dict0:
            if (adv=='霸气'):
                for tmp in other_dict0[adv]:
                    if tmp in query1:
                        return 1
                if adv ==word:
                    return 0
            elif adv=='不':#不 不可以出现在开头和结尾
                if adv==word:
                    if ('不小布' in query1) or((query1[-2]=='不' )&(query1[-1]=='?')) or(query1[-1]=='不')or('闲' in query1):
                        return 1
                    else:
                        return 0
            elif adv=='怎么':
                if (adv==word)&(len(eval(word1))==1):
                    for tmp in other_dict0[adv]:
                        if tmp in query1:
                            return 1
                    return 0
            elif (adv=='前')&(len(eval(word1))==1)&(adv==word):
                return 0
            elif adv in word:
                return 0
                
    return -1

def is_eq_per2(x):
    if (eval(x[3])[0]=='仲')&(eval(x[4])[0]=='钮'):
        return 0
    return -1
def zenm(x):
    dic=load_dict('./work/user_data/stop_data/stopword')
    zenme_dict=dic['zenme_dict']
    if (len(eval(x))==1)&(eval(x)[0] in zenme_dict)or('怎么' in eval(x)) &('现在' not in eval(x))&('颜色' not in eval(x)):
        return 1
    else:
        return 0

#i和o模糊修改
def mohu(x,y):
    x=eval(x)[0]
    y=eval(y)[0]
    for i in zip(x,y):
        if (i==('i','1'))or(i==('1','i'))or(i==('o','0'))or(i==('0','o')):
            return 1
    return -1

#纠正LAC实体名词
def LAC_(x,lactype):
    '''
    修改LAC分词后的ORG实体
    Parameters
    ----------
    x : 数据集

    Returns
    -------
    lca_word1 : query1分词
    lca_a : query1分词词性
    lca_word2 : query2分词
    lca_b : query2分词词性

    '''
    dic=load_dict('./work/user_data/stop_data/stopword')
    if lactype=='ORG1':
        lca_word1=eval(x[2])
        lca_a=eval(x[3])
        lca_word2=eval(x[4])
        lca_b=eval(x[5])
        org=dic['org1']
        for tmp in list(org.keys()):
            if tmp in lca_word1:
                lca_a[lca_word1.index(tmp)]='ORG'
                lca_word1[lca_word1.index(tmp)]=org[tmp]
            if tmp in lca_word2:
                lca_b[lca_word2.index(tmp)]='ORG'
                lca_word2[lca_word2.index(tmp)]=org[tmp]
            
        if ('华夏' in lca_word1)&('航空' not in x[0]):
            lca_a[lca_word1.index('华夏')]='ORG'
            lca_word1[lca_word1.index('华夏')]='华夏银行'
        if ('华夏' in lca_word2)&('航空' not in x[1]):
            lca_b[lca_word2.index('华夏')]='ORG'
            lca_word2[lca_word2.index('华夏')]='华夏银行'
        if ('全国军事类' in x[0]):
            lca_a[lca_word1.index('全国')]='ORG'
            lca_word1[lca_word1.index('全国')]='全国军事大学'
        if ('全国军事类' in x[1]):
            lca_b[lca_word2.index('全国')]='ORG'
            lca_word2[lca_word2.index('全国')]='全国军事大学'
        if ('农科类' in x[0]):
            lca_a[lca_word1.index('农科类')]='ORG'
            lca_word1[lca_word1.index('农科类')]='农科大学'
        if ('农科类' in x[1]):
            lca_b[lca_word2.index('农科类')]='ORG'
            lca_word2[lca_word2.index('农科类')]='农科大学'
        if ('西安文科类' in x[0]):
            lca_a[lca_word1.index('西安')]='ORG'
            lca_word1[lca_word1.index('西安')]='西安文科大学'
        if ('西安文科类' in x[1]):
            lca_b[lca_word2.index('西安')]='ORG'
            lca_word2[lca_word2.index('西安')]='西安文科大学'

    if lactype=='ORG2':
        org=dic['org2']
        lca_a=eval(x[5])
        lca_word1=eval(x[6])
        lca_b=eval(x[7])
        lca_word2=eval(x[8])

        if ('芒果' in lca_word1):
            lca_a[lca_word1.index('芒果')]='n'
        if ('芒果' in lca_word2):
            lca_b[lca_word2.index('芒果')]='n'
        
        for tmp in list(org.keys()):
            if tmp in lca_word1:
                lca_word1[lca_word1.index(tmp)]=org[tmp]
            if tmp in lca_word2:
                lca_word2[lca_word2.index(tmp)]=org[tmp]

            
    if lactype=='PER':
        lca_a=eval(x[5])
        lca_word1=eval(x[6])
        lca_b=eval(x[7])
        lca_word2=eval(x[8])
        per=dic['per']
        for tmp in list(per.keys()):
            if tmp in lca_word1:
                lca_a[lca_word1.index(tmp)]='PER'
                lca_word1[lca_word1.index(tmp)]=per[tmp]
            if tmp in lca_word2:
                lca_b[lca_word2.index(tmp)]='PER'
                lca_word2[lca_word2.index(tmp)]=per[tmp]
        if ('橙子' in lca_word1):
            lca_a[lca_word1.index('橙子')]='n'
        if ('橙子' in lca_word2):
            lca_b[lca_word2.index('橙子')]='n'
            
    if lactype=='LOC':
        lca_a=eval(x[5])
        lca_word1=eval(x[6])
        lca_b=eval(x[7])
        lca_word2=eval(x[8])
        loc=dic['loc']
        for tmp in list(loc.keys()):
            if tmp in lca_word1:
                lca_a[lca_word1.index(tmp)]='LOC'
                lca_word1[lca_word1.index(tmp)]=loc[tmp]
            if tmp in lca_word2:
                lca_b[lca_word2.index(tmp)]='LOC'
                lca_word2[lca_word2.index(tmp)]=loc[tmp]
        if ('J' in lca_word1):
            lca_a[lca_word1.index('J')]='xc'
        if ('J' in lca_word2):
            lca_b[lca_word2.index('J')]='xc'
        

    return lca_word1,lca_a,lca_word2,lca_b

#取实体名词对比
def shouji(x,y):
    dic=load_dict('./work/user_data/stop_data/stopword')
    shouji_dict=dic['shouji_dict']
    x=eval(x)[0]
    for shouji in shouji_dict:
        if (shouji=='的')&(shouji==x):
            if ('什么意思' in y):
                return 0
            else:
                return 1
        if shouji == x:
            return 1
       
    return -1

def is_shuzi(x,y):
    dic=load_dict('./work/user_data/stop_data/stopword')
    if (x==eval(y)[0]):
        return 1
    elif (x in eval(y)[0]):
        for tmp in dic['shuzi']:
            if tmp in y:
                return 0
        return 1
    else:
        return 0
    
#取ORG位置上词对比
import itertools
def is_eq_org1(x):
    lca_word1,lca_a,lca_word2,lca_b=LAC_(x,lactype='ORG1')
    dic=load_dict('./work/user_data/stop_data/stopword')
    
    org_a=[]
    time_a=[]
    loc_a=[]
    m_a=[]

    for a in zip(lca_word1,lca_a):#依次加入词性
        if a[1]=='ORG':
            org_a.append(a[0])
        if a[1]=='TIME':
            time_a.append(a[0])
        if a[1]=='LOC':
            loc_a.append(a[0])
        if a[1]=='m':
            m_a.append(a[0])

    org_b=[]
    time_b=[]
    loc_b=[]
    m_b=[]

    for b in zip(lca_word2,lca_b):#依次加入词性
        if b[1]=='ORG':
            org_b.append(b[0])
        if b[1]=='TIME':
            time_b.append(b[0])
        if b[1]=='LOC':
            loc_b.append(b[0])
        if b[1]=='m':
            m_b.append(b[0])
            

    if (len(org_a)==0 ) or (len(org_b)==0 ):#不存在ORG
        return 0
    elif (len(org_a)==1 ) & (len(org_b)==1 ):#仅存在一个ORG
        if (org_a[0]==org_b[0]):
            for tmp in dic['org_stop']:
                if ((tmp in x[0])&(dic['org_stop'][tmp] in x[1]))or((tmp in x[1])&(dic['org_stop'][tmp] in x[0])):
                    return 0
            if (len(time_a)!=0)&(len(time_b)!=0):#存在时间
                if  (time_a[0]==time_b[0]):
                    return 1
                else:
                    return 0
            if (len(loc_a)!=0)&(len(loc_b)!=0):#存在地点
                if  (loc_a[0]==loc_b[0]):
                    return 1
                else:
                    return 0
            if (len(m_a)!=0)&(len(m_b)!=0):#存在数字
                if  (m_a[0]==m_b[0]):
                    return 1
                else:
                    return 0
            return 1
        else:
            return 0
    elif (len(org_a)==2 ) & (len(org_b)==2 ):#存在两个ORG
        if (('和' in x[0]) &('和' in x[1]))or (('与' in x[0]) &('与' in x[1]))or (('还是' in x[0]) &('还是' in x[1])):#并列关系
            if ((org_a[0]==org_b[0]) & (org_a[1] == org_b[1])) or((org_a[1]==org_b[0]) & (org_a[0] == org_b[1])):
                return 1
            else:
                return 0
        else:
            if (org_a[0]==org_b[0]) & (org_a[1] == org_b[1]):#比较
                return 1
            else:
                return 0
    else:#长度不等
        length=0
        for tup in itertools.zip_longest(org_a,org_b):
            if (tup[0]!=None)&(tup[1]!=None):
                if (tup[0] in x[1])&(tup[1] in x[0]):
                    length+=1
            elif tup[0]!=None:
                if tup[0] in x[1]:
                    length+=1
            elif tup[1]!=None:
                if tup[1] in x[0]:
                    length+=1
        if length==2:
            return 1
        else:
            return 0
        
    return -1

#取PER人名对比
def is_eq_per(x):
    lca_word1,lca_a,lca_word2,lca_b=LAC_(x,lactype='PER')
    dic=load_dict('./work/user_data/stop_data/stopword')
   
    per_a=[]
    for a in zip(lca_word1,lca_a):
        if a[1]=='PER':
            per_a.append(a[0])     
    per_b=[]
    for b in zip(lca_word2,lca_b):
        if b[1]=='PER':
            per_b.append(b[0])

    for per in zip(per_a,per_b):
        for tmp in dic['PER_name']:
            if tmp in x[0]:
                return -1
        if (('和' in x[0])&('和' in x[1])) | (('与' in x[0])&('与' in x[1])):#并列关系
            if len(list(set(per_a).difference(set(per_b))))==0:
                    return 1
        if (per[0]!=per[1]):
            return 0
    return -1

#取ORG实体对比
def is_eq_org2(x):
    lca_word1,lca_a,lca_word2,lca_b=LAC_(x,lactype='ORG2')
    org_a=[]
    for a in zip(lca_word1,lca_a):
        if a[1]=='ORG':
            org_a.append(a[0])
            
    org_b=[]
    for b in zip(lca_word2,lca_b):
        if b[1]=='ORG':
            org_b.append(b[0])
    
    for org in zip(org_a,org_b):
        if (('和' in x[0])&('和' in x[1])) | (('与' in x[0])&('与' in x[1])):#并列关系
            if len(org_a)!=len(org_b):
                return -1
            if len(list(set(org_a).difference(set(org_b))))==0:
                    return 1
        if (org[0]!=org[1])&('OPPO' not in x[0]):#排除手机品牌
            return 0
    return -1

#取LOC实体对比
def is_eq_loc(x):
    lca_word1,lca_a,lca_word2,lca_b=LAC_(x,lactype='LOC')
    dic=load_dict('./work/user_data/stop_data/stopword')
    if 'TIME' in lca_a:#忽略包含时间
        return -1
    loc_a=[]
    for a in zip(lca_word1,lca_a):
        if a[1]=='LOC':
            loc_a.append(a[0])
            
    loc_b=[]
    for b in zip(lca_word2,lca_b):
        if b[1]=='LOC':
            loc_b.append(b[0])
    
    for loc in zip(loc_a,loc_b):
        if ((('和' in x[0])&('和' in x[1])) | (('与' in x[0])&('与' in x[1])))| (('还是' in x[0])&('还是' in x[1]))&('呼和浩特' not in x[0]):#并列关系
            if len(loc_a)!=len(loc_b):
                return -1
            if len(list(set(loc_a).difference(set(loc_b))))==0:
                    return 1
        if ('到' in x[0])|('离' in x[0]):
            if (len(loc_a)==2)&(len(loc_b)==2):
                if ('多远' in x[1])|('公里' in x[1]):
                    if len(list(set(loc_a).difference(set(loc_b))))==0:
                            return 1
                    else:
                        return 0
        if (loc[0]!=loc[1]):#LOC对应位置不同
            return 0
        for tmp in dic['time_freq']:
            if (tmp in x[3])or (tmp in x[4]):
                return 0
    return -1

#TIME
def time1(word1,word2):
    word1=eval(word1)[0]
    word2=eval(word2)[0]
    dic=load_dict('./work/user_data/stop_data/stopword')
    for tmp in list(dic['jinyi_dict'].keys()):
        if ((tmp in word1)&(dic['jinyi_dict'][tmp] in word2)) or((tmp in word2)&(dic['jinyi_dict'][tmp] in word1)):
            return 1
    for tmp in list(dic['fanyi_dict'].keys()):
        if ((tmp in word1)&(dic['fanyi_dict'][tmp] in word2)) or((tmp in word2)&(dic['fanyi_dict'][tmp] in word1)):
            return 0
    for adv in dic['time1_dict']:
        if ((adv in word1)&(adv not in word2))or ((adv in word2)&(adv not in word1)):
            return 0
    return -1


def time2(word1,word2):
    word1=eval(word1)[0]
    word2=eval(word2)[0]
    dic=load_dict('./work/user_data/stop_data/stopword')
    for adv in dic['time2_dict']:
        if ((adv in word1)&(adv not in word2))|((adv in word2)&(adv not in word1)):
            if (adv=='最近')&(('这两天'==word1)|(('这两天'==word2))):
                return 1
            elif (word1=='早')&(word2=='前'):
                return -1
            return 0
    return -1

def time_(x):
    word1=eval(x[3])[0]
    word2=eval(x[4])[0]
    dic=load_dict('./work/user_data/stop_data/stopword')
    for tmp in list(dic['jinyi_dict'].keys()):
        if ((tmp in word1)&(dic['jinyi_dict'][tmp] in word2)) or((tmp in word2)&(dic['jinyi_dict'][tmp] in word1)):
            return 1
    for tmp in list(dic['fanyi_dict'].keys()):
        if ((tmp in word1)&(dic['fanyi_dict'][tmp] in word2)) or((tmp in word2)&(dic['fanyi_dict'][tmp] in word1)):
            return 0
    for adv in dic['time_dict']:
        if ((adv in word1)&(adv not in word2))|((adv in word2)&(adv not in word1)):
            if ((('现在' in word1)&('现在' in x[1]))|(('现在' in word2)&('现在' in x[0]))|((adv=='现在')&(('目前' in x[0])|('目前' in x[1]))))&('LOC' not in eval(x[5])):
                return 1
            return 0
    return -1