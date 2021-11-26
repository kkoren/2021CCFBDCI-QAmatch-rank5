import sys
sys.path.append('/home/aistudio/external-libraries')
import random
import pandas as pd
from pyhanlp import HanLP

random.seed(2021)

def get_keyword(content, keynum=2):
    """
    获取每个问题中的关键字,关键词的数目由keynum控制
    :param content: 一个句子
    :return:
    """
    keywordList = HanLP.extractKeyword(content, keynum)
    return keywordList

def construct_synwords(cilinpath='./work/user_data/eda_data/cilin.txt'):
    """
    根据哈工大的同义词词林（cilin.txt）构建同义词表，文件来自https://github.com/TernenceWind/replaceSynbycilin/blob/master/cilin.txt
    :param cilinpath:  同义词词林的路径
    :return:
    """
    synwords = []
    with open(cilinpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip()
            split = temp.split(' ')
            bianhao = split[0]
            templist = split[1:]
            if bianhao[-1] == '=':
                synwords.append(templist)
    return synwords

def replace_synwords(content,synwords):
    """
    使用同义词替换content中的关键词
    :param content:  需要进行同义词替换的句子，不是整个样本或者数据集
    :param synwords: 同义词词典
    :return:
    """
    segmentationList = HanLP.segment(content)
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content,keynum)   # 获取关键词

    segmentationList = [term.word for term in segmentationList]
    replace_word = {}
    #查询content中的关键词在同义词表中的近义词
    for word in keywordList:
        if word in segmentationList:
            for syn in synwords:
                # if word in syn:   # 设计替换规则
                if word == syn[0]:
                    if len(syn) == 1:
                        continue
                    else:
                        # 以最靠近word的词替换word
                        if syn.index(word) == 0:
                            replace_word[word] = (syn[1])
                        else:
                            replace_word[word] = (syn[syn.index(word)-1])
                else:
                    continue
        else:
            continue

    # 替换content中的同义词
    for i in range(len(segmentationList)):
        if segmentationList[i] in replace_word:
            segmentationList[i] = replace_word[segmentationList[i]]
        else:
            continue
    # 将content重新组合成句子
    content_new = "".join(segmentationList)
    # 返回经过替换后的content,即new_content
    return content_new



def get_same_pinyin_vocabulary(same_pinyin_file):
    """
    获得具有相同拼音的词表，文件来自https://github.com/shibing624/pycorrector/blob/master/pycorrector/data/same_pinyin.txt
    :param same_pinyin_file:
    :return: {"word1":samepinyin,"word2":samepinyin}
    """
    same_pinyin = {}
    with open(same_pinyin_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.strip("\n")
            split_1 = temp.split('\t')
            word_index = split_1[0]  # 词根
            # word_same_1 = split_1[1]   #同音同调
            # word_same_2 = split_1[2]   #同音异调
            # word_samePinyin = split_1[1]+split_1[2]
            sameWords = ""
            for i in split_1[1:]:  # 拼接同音同调和同音异调词表
                sameWords += i
            same_pinyin[word_index] = list(sameWords)  # 将同音同调和同音异调放在同一个list中
            # same_pinyin[word_index] = [list(word_same_1),list(word_same_2)]   # 将同音同调和同音异调放在不同list中
    # 格式[word,freq]
    return same_pinyin


def get_word_freq(chinese_word_freq_file_path):
    '''
    读取word,frequency ,构建词典
    :param chinese_word_freq_file_path:   中文词频文件
    :return: {"word1":freq1,"word2":freq2}
    '''
    word_freq_vocab = {}  # 词频字典,格式为[“word”:freq]
    with open(chinese_word_freq_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            word_freq = line.split(" ")
            if word_freq[0] not in word_freq_vocab:
                word_freq_vocab[word_freq[0]] = int(word_freq[1])  # 添加"word"：freq到词典中
            else:
                pass
    # print("word_freq_vocab", word_freq_vocab["火"])
    return word_freq_vocab


def replace_samePinyin(content, same_pinyin, word_freq_vocab, replace_num=1):
    """
    使用同音字替换content中关键词中，（替换规则为替换掉所有同音字出现频率最高的那个字）
    :param content:  要替换的文本
    :param same_pinyin: 相同拼音词汇表
    :param word_freq_vocab: 汉语字频率表
    :param replace_num: 要替换的数量，这里只替换一个字
    :return: 经过相同拼音替换掉的文本
    """
    segmentationList = HanLP.segment(content)
    word_list_of_content = list(content)
    # print(len(segmentationList))
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content, keynum)  # 获取关键词
    key_character = []
    for word in keywordList:  # 提取关键词里的关键字
        key_character += list(word)
    key_character = list(set(key_character))  # 去掉重复的关键字
    key_character = [word for word in key_character if word in same_pinyin]  # 先检查关键词中的所有字是否都出现在same_pinyin词汇表中
    word_freq = []
    for i in key_character:  # 统计关键字的频率
        samePinyin_list = same_pinyin[i]  # 获取相同拼音的所有字
        samePinyin_freq = []
        for j in samePinyin_list:
            if j in word_freq_vocab:
                samePinyin_freq.append(word_freq_vocab[j])
            else:
                samePinyin_freq.append(1)
        word_freq.append(samePinyin_list[samePinyin_freq.index(max(samePinyin_freq))])
    freq = []
    if len(word_freq) != 0:
        for i in word_freq:
            if i in word_freq_vocab:
                freq.append(word_freq_vocab[i])
            else:
                freq.append(1)
        same_pinyin_HighFreq_word = word_freq[freq.index(max(freq))]
        replace_word = key_character[freq.index(max(freq))]
        replace_index = word_list_of_content.index(replace_word)
        word_list_of_content[replace_index] = same_pinyin_HighFreq_word
        new_content = "".join(word_list_of_content)
        # print("smae_pinyin",same_pinyin["火"])
        return new_content
    else:
        return content



def read_csvToDF(data_path):
    data = []
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_data=[i for i in line.rstrip().split("\t")]
            if len(tmp_data) != 3:
                continue
            data.append(tmp_data)
                
    df=pd.DataFrame(data,columns=['text_one','text_two','label'])
    #调整label类型为int
    df['label']= df['label'].astype(int)
    return df

def synword_and_samepinyin_data(data,cilin_path,same_pinyin_file,chinese_word_freq_file,replace_rule=True, 
columns_num=3, word_portition=0.2,pinyin_portition=0.15):
    """
    从data中取一定比例的样本，进行数据增强
    :param data: 数据集
    :param save_data_file:  保存增强后的数据集的文件
    :param cilinpath: 近义词的词表文件
    :param same_pinyin_file:  同音字的词表文件
    :param chinese_word_freq_file:  中文字频文件
    :param repalce_rule: True为从正样本中的q1和q2中随机选择一个作为待替换的文本
                        false则将q1和q2都进行同义词替换
    :param portition: 要替换的样本的比例
    :return: 经过数据增强后的数据集
    """

    synonyms_list=construct_synwords(cilin_path) # 获取近义词词表
    same_pinyin_vocab = get_same_pinyin_vocabulary(same_pinyin_file)  # 获取相同拼音的词表
    word_freq_vocab = get_word_freq(chinese_word_freq_file)  # 获取汉字词频表
    if word_portition != 1:
        word_samples = data.sample(frac=word_portition, replace=False, random_state=2021)  # 随机选一定比例的样本进行数据增强
    else:
        word_samples = data_df

    word_samples['text_one_word']=word_samples['text_one'].apply(lambda x:replace_synwords(x,synonyms_list))
    word_samples['text_two_word']=word_samples['text_two'].apply(lambda x:replace_synwords(x,synonyms_list))

    if pinyin_portition != 1:
        pinyin_sample = data.sample(frac=pinyin_portition, replace=False, random_state=1998)  # 随机选一定比例的样本进行数据增强
    else:
        pinyin_sample = data_df

    pinyin_sample['text_one_pinyin']=pinyin_sample['text_one'].apply(lambda x:replace_samePinyin(x,same_pinyin_vocab,word_freq_vocab))
    pinyin_sample['text_two_pinyin']=pinyin_sample['text_two'].apply(lambda x:replace_samePinyin(x,same_pinyin_vocab,word_freq_vocab))

    return word_samples,pinyin_sample


from sklearn.utils import shuffle
def EDA_data(wordtmp,pinyintmp):
    '''
    拼接同义词增强和同音同调同音异调粗词的数据集
    '''
    wordtmp['word_similar_one']=list(map(lambda x,y:0 if x==y else 1,wordtmp['text_one'],wordtmp['text_one_word']))
    wordtmp['word_similar_two']=list(map(lambda x,y:0 if x==y else 1,wordtmp['text_two'],wordtmp['text_two_word']))
    wordtmp['similar_word']=list(map(lambda x,y:1 if x==1 or y==1 else 0,wordtmp['word_similar_one'],wordtmp['word_similar_two']))
    print(len(wordtmp[wordtmp['similar_word']==1]))
    word_eda=wordtmp[wordtmp['similar_word']==1]
    word_eda=word_eda[['text_one_word','text_two_word','label']]
    word_eda.rename(columns={'text_one_word':'text_one','text_two_word':'text_two'},inplace=True)
    pinyintmp['pinyin_similar_one']=list(map(lambda x,y:0 if x==y else 1,pinyintmp['text_one'],pinyintmp['text_one_pinyin']))
    pinyintmp['pinyin_similar_two']=list(map(lambda x,y:0 if x==y else 1,pinyintmp['text_two'],pinyintmp['text_two_pinyin']))
    pinyintmp['similar_pinyin']=list(map(lambda x,y:1 if x==1 or y==1 else 0,pinyintmp['pinyin_similar_one'],pinyintmp['pinyin_similar_two']))
    #print(len(pinyintmp[pinyintmp['similar_pinyin']==1]))
    pinyin_eda=pinyintmp[pinyintmp['similar_pinyin']==1]
    pinyin_eda=pinyin_eda[['text_one_pinyin','text_two_pinyin','label']]
    pinyin_eda.rename(columns={'text_one_pinyin':'text_one','text_two_pinyin':'text_two'},inplace=True)
    eda_data=pd.concat([word_eda,pinyin_eda])
    #print(len(eda_data[eda_data['label']==1]))
    #print(len(eda_data[eda_data['label']==0]))
    return eda_data

def random_change(data):
    '''
    对拼接后的增强数据集进行50%随机交换提高模型鲁棒性
    '''
    data=shuffle(data,random_state=2021)
    changedata=data[0:len(data)//2]
    #print(len(changedata))
    source_data=data[len(data)//2:]
    #print(len(source_data))
    tmp=pd.DataFrame()
    
    tmp['text_one']=changedata['text_two']
    tmp['text_two']=changedata['text_one']
    tmp['label']=changedata['label']
    
    final_data=pd.concat([tmp,source_data])
    return final_data

def read_csvToDF(data_path):
    data = []
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_data=[i for i in line.rstrip().split("\t")]
            if len(tmp_data) != 3:
                continue
            data.append(tmp_data)
                
    df=pd.DataFrame(data,columns=['text_one','text_two','label'])
    #调整label类型为int
    df['label']= df['label'].astype(int)
    return df

if __name__ == "__main__":
    train=read_csvToDF('./work/raw_data/train.txt')
    gaiic=read_csvToDF('./work/user_data/eda_data/gaiic_track3_round1_train_20210220.tsv')
    data=pd.concat([train,gaiic])
    cilin_path='./work/user_data/eda_data/cilin.txt'
    same_pinyin_file='./work/user_data/eda_data/same_pinyin.txt'
    chinese_word_freq_file='./work/user_data/eda_data/chinese-words.txt'
    word_data,pinyin_data=synword_and_samepinyin_data(data,cilin_path,same_pinyin_file,chinese_word_freq_file,replace_rule=True,
    columns_num=3, word_portition=0.2,pinyin_portition=0.15)
    word_data.to_csv('./work/user_data/eda_data/word_data.txt')
    pinyin_data.to_csv('./work/user_data/eda_data/pinyin_data.txt')

    eda_data=EDA_data(word_data,pinyin_data)
    final_data=random_change(eda_data)
    final_data.to_csv('./work/user_data/eda_data/word_0.2pinyin_0.15_1.txt',sep='\t',header=None,index=False)


    #读取训练集数据将增强后的文本和原始数据进行拼接
    train=read_csvToDF('./work/raw_data/train.txt')
    gaiic=read_csvToDF('./work/user_data/eda_data/gaiic_track3_round1_train_20210220.tsv')
    eda=read_csvToDF(data_path='./work/user_data/eda_data/word_0.2pinyin_0.15.txt')
    gaiic_train_eda=pd.concat([gaiic,train,eda])
    gaiic_train_eda.to_csv('./work/user_data/eda_data/gaiic_train_eda.txt',sep='\t',index=False,header=None)





