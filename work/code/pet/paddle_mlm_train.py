# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Christina
"""

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from paddlenlp.transformers import *
from paddle_mlm_data import*
from paddle_mlm_model import *

from sklearn.model_selection import KFold



# yapf: disable
#参数
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", type=str, default='train.txt', help="The full path of train_set_file")
parser.add_argument("--dev_set", type=str, default='dev.txt', help="The full path of dev_set_file")
parser.add_argument("--save_checkpoints_name", default='checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=68, type=int, help="The maximum total input sequence length after tokenization. "
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
                    filename='pet_mlm_train.log',
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
    
    
logging.info('------------------------------PET训练--------------------------------')
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
def evaluate(model, criterion, metric, data_loader,vocab_size):
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

    for batch in data_loader:
        input_ids,token_type_ids,target_ids,labels = batch
        
        #计算验证总个数
        total_num += len(labels)
        
        #返回logits1预测值, kl_loss损失
        prediction_scores, seq_relationship_score, kl_loss = model(input_ids=input_ids,token_type_ids=token_type_ids, do_evaluate=True)

        hen_logits=prediction_scores[:,1,pos_id]
        hen_logits=paddle.reshape(hen_logits,[-1,1])
        bu_logits=prediction_scores[:,1,neg_id]
        bu_logits=paddle.reshape(bu_logits,[-1,1])

        logits=paddle.concat([bu_logits,hen_logits],axis=-1)

        # loss=criterion(paddle.reshape(prediction_scores,[-1,vocab_size]), paddle.reshape(target_ids,[-1]))
        loss=criterion(logits,labels)
        losses.append(loss.numpy())

        correct = metric.compute(logits, labels)#acc
        metric.update(correct)#更新metric
        accu = metric.accumulate()

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    logging.info("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    
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
    train_ds = load_dataset(read_text_pair, data_path=args.train_set, is_test=False, lazy=False)#train_ds是MapDataset,train[0]是dict
    # 加载验证集
    dev_ds = load_dataset(read_text_pair, data_path=args.dev_set, is_test=False, lazy=False)
    
    # #将MapDataset转为df
    # train_ds=pd.DataFrame(list(train_ds.data), columns=['query1', 'query2','label'])
    
    
    # # 模拟标注和非标注数据
    # train_frac = 0.01  # 标注数据的比例
    # num_labeled = int(len(train_data) * train_frac)
    # unlabeled_data = data_df[num_labeled:][['query1','query2']]#非标注数据
    # unlabeled_data['label']=2
    # train_data = train_data[:num_labeled]
    # train_ds =pd.concat([train_data,unlabeled_data],axis=0)
    # print('train_ds:',train_ds.shape)
    
    # #再将train_ds转为MapDataset
    # train_ds=train_ds.to_dict(orient='records')
    # train_ds=MapDataset(train_ds)
    
    # 选择模型
    if args.model_name=='ernie-gram':
        print('---------------choose ernie-gram-------------------')
        pretrained_model = ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
        
    if args.model_name=='roberta':
        pretrained_model = RobertaModel.from_pretrained('roberta-wwm-ext')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        
    
    #包装转换函数，返回input_ids（分词）, token_type_ids（分句）
    trans_func = partial(convert_query_to_ids,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length, )

    batchify_fn = lambda samples, fn=Tuple(
        #填充token
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # source_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_id
        Pad(axis=0, pad_val=tokenizer.pad_token_id), # target_ids
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    # 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
    # 长度是指的： 一个batch中的最大长度，主要考虑性能开销
    # paddlenlp.data.Tuple	将多个batchify函数包装在一起
    
    #创造dataloader，返回paddle.io.DataLoader类型
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)


    # 对应的任务描述
    prefix = u'很相似句子对：'
    mask_idx=1 #掩盖pattern中字的位置,因为还有cls所以为1
    global pos_id,neg_id
    pos_id = tokenizer(u'很')['input_ids'][1]#取很的id
    neg_id = tokenizer(u'不')['input_ids'][1]#取不的id


    #构建模型
    model = QuestionMatching_mlm(pretrained_model)

    #设置模型路径
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    
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
    
    
    #计算输入input和标签label间的交叉熵损失，指定一个忽略的标签值
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

            input_ids,token_type_ids,target_ids,label = batch

            #model返回mask预测值,上下句预测和损失
            #prediction_scores大小shape维度为[batch_size, seq_len, vocab_size]，对应句子每一个位置上每个词的概率
            prediction_scores, seq_relationship_score = model(input_ids=input_ids,token_type_ids=token_type_ids)
            
 
            #计算交叉熵
            #预测值为[(batch_size*seq_length), vocab_size],target为[(batch_size*seq_length)]
            #对于输入，每个单词对应一个vocab_size向量,此向量上的每个值就是对该词的预测。例如 x = [23,45,...,65],x[0]=23就是预测x单词是0号单词的评分为23
            #target，为一个[batch_size*seq_length]的一维向量，此向量的值，必须在0-vocab_size中选择（因为每个词都属于词表）
            
            # ce_loss=criterion(paddle.reshape(prediction_scores,[-1,pretrained_model.config["vocab_size"]]), paddle.reshape(target_ids,[-1]))
 
            hen_logits=prediction_scores[:,1,pos_id]
            hen_logits=paddle.reshape(hen_logits,[-1,1])
            bu_logits=prediction_scores[:,1,neg_id]
            bu_logits=paddle.reshape(bu_logits,[-1,1])

            logits=paddle.concat([bu_logits,hen_logits],axis=-1)

            loss=criterion(logits, label)
            
            #计算预测与label的Accuracy
            correct = metric.compute(logits, label)#计算预测与真实损失acc
            metric.update(correct)#更新metric的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前step的准确率
            acc = metric.accumulate()#累积的统计指标，计算和返回准确率。

           
            
            global_step += 1
            if global_step % 10 == 0 and rank == 0:#每十步打印训练损失
                print(
                    "global step %d / %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s"
                    % (global_step, num_training_steps,epoch, step, loss, ce_loss, kl_loss, acc,
                       10 / (time.time() - tic_train)))
                
                logging.info(
                    "global step %d / %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s"
                    % (global_step,num_training_steps, epoch, step, loss, ce_loss, kl_loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            

            loss.backward()#反向传播，计算当前梯度
            optimizer.step()#根据梯度更新网络参数
            lr_scheduler.step()#调整学习率
            optimizer.clear_grad()#清空过往梯度

            if global_step % args.eval_step == 0 and rank == 0:#每十步打印
                #验证集的acc
                accuracy = evaluate(model, criterion, metric, dev_data_loader,pretrained_model.config["vocab_size"])
                
                #保存最好的模型
                if accuracy > best_accuracy:
                    save_dir = os.path.join('./model_data/'+args.model_name+'_pet/'+args.save_checkpoints_name, "model_%d_%.5f" % (global_step,accuracy))
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