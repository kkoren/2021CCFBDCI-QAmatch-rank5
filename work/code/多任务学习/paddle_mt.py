# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christina
"""

'''
多任务训练
'''

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import pandas as pd
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.datasets import MapDataset
# from paddlenlp.transformers import LinearDecayWithWarmup

from paddlenlp.transformers import *
from paddle_data import *

from paddle_model import *

from sklearn.model_selection import KFold
from itertools import zip_longest
# yapf: disable
#参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path_list", default=['LCQMC/','BQ/','OPPO/'], nargs='+', type=str, help="Dataset path list.")
parser.add_argument("--dataset_dev_path_list", default=['LCQMC/','BQ/','OPPO/'], nargs='+', type=str, help="Dataset path list.")
parser.add_argument("--save_checkpoints_name", default='checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=62, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=3E-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=1E-3, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=1, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=2021, help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of" 
    "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--model_name", default='ernie-gram', type=str, help="choice for model" )

args = parser.parse_args()
# yapf: enable


import logging
#日志
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='mutli-mask_base.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

#设置随机数种子
def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

logging.info('------------------------------多任务训练--------------------------------')
#打印所有参数
print('---------------参数配置-----------------')
logging.info('---------------参数配置-----------------')
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
    logging.info('{} : {}'.format(k,(str(args.__dict__[k]))))
logging.info('--------------------------------')
print('--------------------------------')



# 创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有stop_gradient=True。
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0
    batch_logits = []
    for batch in data_loader:
        dataset_id,input_ids,token_type_ids, labels = batch
        
        #计算验证总个数
        total_num += len(labels)
        
        #返回logits1预测值, kl_loss损失
        logits = model(input_ids=input_ids,
            token_type_ids=token_type_ids,
            do_evaluate=True)
        
        batch_logits.append(logits.numpy())

        loss = criterion(logits, labels)#交叉熵损失
        losses.append(loss.numpy())

        correct = metric.compute(logits, labels)#acc
        metric.update(correct)#更新metric
        accu = metric.accumulate()

    #总预测拼接
    batch_logits = np.concatenate(batch_logits, axis=0)

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    logging.info("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))

    model.train()
    metric.reset()

    return batch_logits,accu



def main():
  
    # 设置设备
    paddle.set_device(args.device)
    
    # 多卡训练时当前进程的rank
    rank = paddle.distributed.get_rank()
    
    #动态图
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        
    # 设置种子
    set_seed(args.seed)
    
    # 加载三个训练集，加上dataset编号
    train_dataset_list = [load_dataset(
        read_dataset_mt, data_path= os.path.join(path, "train.txt"),dataset_id=i, is_test=False, lazy=False) for i, path in enumerate(args.dataset_path_list)]

    # 加载三个验证集
    dev_dataset_list = [load_dataset(
        read_dataset_mt, data_path= os.path.join(path, "dev"),dataset_id=i,is_test=False, lazy=False) for i,path in enumerate(args.dataset_dev_path_list)]
 

    #统计每个数据集中label的类别数
    labels_num_list=[ 2 for _ in args.dataset_path_list]


    # 选择模型
    if args.model_name=='ernie-gram':
        print('---------------choose ernie-gram-------------------')
        pretrained_model = ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
        
    if args.model_name=='roberta':
        pretrained_model = RobertaModel.from_pretrained('roberta-wwm-ext')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        
    
    
    #包装转换函数，返回input_ids（分词）, token_type_ids（分句）
    trans_func = partial(
        convert_query_to_ids_datasetid,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Stack(dtype="int64"),#数据集的序号
        #填充token
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # 填充句子对
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # 填充分割符号
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    # 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
    # 长度是指的： 一个batch中的最大长度，主要考虑性能开销
    # paddlenlp.data.Tuple	将多个batchify函数包装在一起


    #打乱数据集
    # random.shuffle(train_dataset_list_all)
    # train_ds=MapDataset(train_dataset_list_all)

    #创造dataloader，返回paddle.io.DataLoader类型
    train_data_loader_list= [create_dataloader(dataset, mode='train',batch_size=args.train_batch_size, batchify_fn=batchify_fn,trans_fn=trans_func)
                         for  _,dataset in enumerate(train_dataset_list)]#打乱

    dev_data_loader_list= [create_dataloader(dataset, mode='dev',batch_size=args.train_batch_size, batchify_fn=batchify_fn,trans_fn=trans_func) 
                             for  _,dataset in enumerate(dev_dataset_list)]#不打乱
    

    #构建多任务模型
    model = MultitaskClassifier(pretrained_model,dropout=0.15,labels_num_list=labels_num_list)

    #设置模型参数加载路径
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    
    #通过数据并行模式执行动态图模型
    #使用 paddle.distributed.launch 方法启动
    # model = paddle.DataParallel(model)
    
    #总训练步数
    num_training_steps =sum([len(dataset) for dataset in train_data_loader_list])* args.epochs
    print('=============num_training_steps:',num_training_steps)
    
     # 学习率预热比例LinearDecayWithWarmup
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, 
                                         num_training_steps,
                                         args.warmup_proportion)
    
    # 损失参数
    no_decay=["bias", "norm"]
    decay_params = [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
    ]

    # AdamW优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    
    #计算输入input和标签label间的交叉熵损失 
    criterion = paddle.nn.loss.CrossEntropyLoss()
    
    #评估
    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    
    #开始时间
    tic_train = time.time()

    print('LCQMC:',len(train_data_loader_list[0]))
    print('BQ:',len(train_data_loader_list[1]))
    print('OPPO:',len(train_data_loader_list[2]))

    #遍历epoch（将训练集数据全部训练）
    for epoch in range(1, args.epochs + 1):
        model.train()
        #batch，将一批数据对模型权重进行一次反向传播的参数更新
        for step, batch in enumerate(zip(train_data_loader_list[0],train_data_loader_list[1],train_data_loader_list[2]), start=1):
            if batch[0]:
                global_step,tic_train=train_model_dataset('LCQMC',batch[0],model,metric,criterion,lr_scheduler,global_step,num_training_steps,optimizer,epoch,step,tic_train)
            if batch[1]:
                global_step,tic_train=train_model_dataset('BQ',batch[1],model,metric,criterion,lr_scheduler,global_step,num_training_steps,optimizer,epoch,step,tic_train)
            if batch[2]:
                global_step,tic_train=train_model_dataset('OPPO',batch[2],model,metric,criterion,lr_scheduler,global_step,num_training_steps,optimizer,epoch,step,tic_train)
            
            if global_step % args.eval_step == 0:#每一百步验证
                total_acc=0

                #验证集的acc
                for dataset_id, path in enumerate(args.dataset_path_list):
                    dev_data_loader=dev_data_loader_list[dataset_id]
                    model.change_dataset(dataset_id)
                    #验证集的acc
                    _,accuracy = evaluate(model, criterion, metric, dev_data_loader)
                    total_acc+=accuracy
                
                
                #保存最好的模型
                if total_acc/3 > best_accuracy:
                    
                    save_dir = os.path.join('./model_data/'+args.model_name+'_mutlimask_model_power/'+args.save_checkpoints_name, "model_%d_%.5f" % (global_step,total_acc/3))
                    print('--------------save best model----------------')
                    print(save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)#路径不存在，创建路径
                    
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    
                    paddle.save(model.state_dict(), save_param_path)#保存参数
                    tokenizer.save_pretrained(save_dir)

                    best_accuracy = total_acc/3 #赋值最好acc

                if global_step ==  args.max_steps:
                    return

            
    

def train_model_dataset(train_name,batch,model,metric,criterion,lr_scheduler,global_step,num_training_steps,optimizer,epoch,step,tic_train):

    dataset_id,input_ids,token_type_ids, labels =batch

    #设置dataset_id
    model.change_dataset(int(dataset_id[0]))
    
    #model返回预测值和损失
    logits1 = model(input_ids=input_ids,token_type_ids=token_type_ids)
    
   
    #计算预测与label的Accuracy
    correct = metric.compute(logits1, labels)#计算预测与真实损失acc
    metric.update(correct)#更新metric的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前step的准确率
    acc = metric.accumulate()#累积的统计指标，计算和返回准确率。

    loss=criterion(logits1, labels)#交叉熵
    global_step += 1
   
    print(
        "train_name: %s, global step %d /%d, epoch: %d, batch: %d, loss: %.4f, accu: %.4f, speed: %.2f step/s"
        % (train_name,global_step,num_training_steps, epoch, step, loss, acc,
        10 / (time.time() - tic_train)))

    logging.info("train_name: %s, global step %d /%d, epoch: %d, batch: %d, loss: %.4f, accu: %.4f, speed: %.2f step/s"
        % (train_name,global_step,num_training_steps, epoch, step, loss, acc,
        10 / (time.time() - tic_train)))
    tic_train = time.time()
    
    
    loss.backward()#反向传播，计算当前梯度
    optimizer.step()#根据梯度更新网络参数
    lr_scheduler.step()#调整学习率
    optimizer.clear_grad()#清空过往梯度

    return global_step,tic_train



if __name__ == "__main__":
    main()  
