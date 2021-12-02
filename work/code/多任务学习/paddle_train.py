# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christian
"""

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



# yapf: disable
#参数
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", type=str, default='train.txt', help="The full path of train_set_file")
parser.add_argument("--dev_set", type=str, default='dev.txt', help="The full path of dev_set_file")
parser.add_argument("--save_checkpoints_name", default='checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=62, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=3E-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=1E-3, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
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
                    filename='mutli-mask_finetuning_duan_power.log',
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

logging.info('------------------------------多任务微调训练--------------------------------')
#打印所有参数
print('---------------参数配置-----------------')
logging.info('---------------参数配置-----------------')
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
    logging.info('{} : {}'.format(k,(str(args.__dict__[k]))))
logging.info('--------------------------------')
print('--------------------------------')


def search_best_split(dev_logits,dev_label,metric):

    metric.reset()
    print('-------------------------------正在阈值搜索--------------------------------')
    logging.info('-------------------------------正在阈值搜索--------------------------------')
    

    #在验证集上搜索
    i_bst = 0
    bst = 0
    
    dev_logits=pd.DataFrame(dev_logits,columns=['dev_logits0','dev_logits1'])

    dev_label=paddle.to_tensor(dev_label)#转换形式

    print(dev_logits['dev_logits1'])
    for i in np.arange(0,1, 0.05):
      
        dev_logits['dev_logits1']=dev_logits['dev_logits1'].apply(lambda x: 1 if x >=i else 0)#在1概率上寻找

        print(dev_logits['dev_logits1'])
        print(len(dev_logits[dev_logits['dev_logits1']==1]))

        logits_tensor=np.array(dev_logits)
        logits_tensor=paddle.to_tensor(logits_tensor)#转换形式

        correct = metric.compute(logits_tensor, dev_label)#acc
        metric.update(correct)#更新metric
        tra_accu = metric.accumulate()

        print('阈值：{},,accuracy: {}'.format(i, tra_accu))
        logging.info('阈值：{}, accuracy: {}'.format(i, tra_accu))

        if tra_accu> bst:
            i_bst = i
            bst = tra_accu
        
        metric.reset()
    
    print('best split point: {}, best acc: {}'.format(i_bst, bst))
    logging.info('best split point: {}, best acc: {}'.format(i_bst, bst))
    print('----------------------------------------------------------------------------------')
    logging.info('----------------------------------------------------------------------------------')

    metric.reset()
    return i_bst


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
    label=[]
    batch_logit=[]
    for batch in data_loader:
        input_ids,token_type_ids, labels = batch
        
        #计算验证总个数
        total_num += len(labels)
        
        #返回logits1预测值, kl_loss损失
        logits= model(input_ids=input_ids,
            token_type_ids=token_type_ids,
            do_evaluate=True)
        
        loss = criterion(logits, labels)#交叉熵损失
        losses.append(loss.numpy())


        correct = metric.compute(logits, labels)#acc
        metric.update(correct)#更新metric
        accu = metric.accumulate()

        batch_logit.append(logits.numpy())#预计值
        label.append(labels.numpy())#加入label

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    logging.info("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))

    #总预测
    batch_logits = np.concatenate(batch_logit, axis=0)
    batch_labels = np.concatenate(label, axis=0)
    # ibs=search_best_split(batch_logits,batch_labels,metric)

    model.train()
    metric.reset()
    return accu


def do_train_1fold():
    # 设置设备
    paddle.set_device(args.device)
    
    # 多卡训练时当前进程的rank
    rank = paddle.distributed.get_rank()
    
    #动态图
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        
    # 设置种子
    set_seed(args.seed)

    # 加载训练集
    train_ds = load_dataset(
        read_text_pair, data_path=args.train_set, is_test=False, lazy=False)#含有软标签

    # 加载验证集
    dev_ds = load_dataset(
        read_text_pair, data_path=args.dev_set, is_test=False, lazy=False)
    
    # 选择模型
    if args.model_name=='ernie-gram':
        print('---------------choose ernie-gram-------------------')
        pretrained_model = ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
        
    if args.model_name=='roberta':
        pretrained_model = RobertaModel.from_pretrained('roberta-wwm-ext')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
            

                                                                                     
    #包装转换函数，返回input_ids（分词）, token_type_ids（分句）
    trans_func= partial(
        convert_query_to_ids,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        )

    batchify_fn = lambda samples, fn=Tuple(
        #填充token
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input1
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment2
        Stack(dtype="int64"),  # label
    ): [data for data in fn(samples)]



    #创造dataloader，返回paddle.io.DataLoader类型
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func,
       )

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    #构建模型
    model = Classifier(pretrained_model, rdrop_coef=args.rdrop_coef)

    #设置模型路径
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
        
    

    #通过数据并行模式执行动态图模型
    #使用 paddle.distributed.launch 方法启动
    model = paddle.DataParallel(model)
    
    #总训练步数
    num_training_steps = len(train_data_loader) * args.epochs
    print('num_training_steps:',num_training_steps)


    # 学习率预热比例LinearDecayWithWarmup
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, 
                                         num_training_steps,
                                         args.warmup_proportion)
    
    # 损失参数
    #model.named_parameters()返回model的所有参数的(name, tensor)的键值对
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
    
    #遍历epoch（将训练集数据全部训练）
    for epoch in range(1, args.epochs + 1):
        #batch，将一批数据对模型权重进行一次反向传播的参数更新
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids,token_type_ids, labels = batch
            
            #model返回预测值和损失
            logits1 = model(input_ids=input_ids,token_type_ids=token_type_ids)
            
            #损失
            loss=criterion(logits1,labels)

            #计算预测与label的Accuracy
            correct = metric.compute(logits1, labels)#计算预测与真实损失acc
            metric.update(correct)#更新metric的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前step的准确率
            acc = metric.accumulate()#累积的统计指标，计算和返回准确率。

            

            global_step += 1
            if global_step % 10 == 0 and rank == 0:#每十步打印训练损失
                print(
                    "global step %d / %d, epoch: %d, batch: %d, loss: %.4f, accu: %.4f, speed: %.2f step/s"
                    % (global_step,num_training_steps ,epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                logging.info("global step %d / %d, epoch: %d, batch: %d, loss: %.4f, accu: %.4f, speed: %.2f step/s"
                    % (global_step,num_training_steps ,epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            
            
            loss.backward()#反向传播，计算当前梯度
            optimizer.step()#根据梯度更新网络参数
            lr_scheduler.step()#调整学习率
            optimizer.clear_grad()#清空过往梯度

            if global_step % args.eval_step == 0 and rank == 0:#每一百步验证

                #验证集的acc
                accuracy = evaluate(model, criterion, metric, dev_data_loader)
                
                #保存最好的模型
                if accuracy > best_accuracy:
                    save_dir = os.path.join('./model_data/'+args.model_name+'_mt_final_power_quan/'+args.save_checkpoints_name, "model_%d_%.5f" % (global_step,accuracy))

                    print(save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
                    best_accuracy = accuracy

            if global_step ==  args.max_steps:
                return
            
if __name__ == "__main__":
    do_train_1fold()         