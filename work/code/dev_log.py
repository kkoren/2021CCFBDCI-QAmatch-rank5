from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import *

from data import create_dataloader, read_text_pair, convert_example
from GRU_pool_model import QuestionMatching




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
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default='./work/user_data/gaiic_eda_real_pool_gru_droo0.4+0.2_checkpoints/model_17800_acc_0.8805/model_state.pdparams', help="The path of checkpoint to be loaded.")
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
                    filename='work/log/dev_predcit.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

logging.info('------------------------------验证集效果--------------------------------')


#设置随机数种子
def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

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

    for batch in data_loader:
        input_ids,token_type_ids, labels = batch
        
        #计算验证总个数
        total_num += len(labels)
        
        #返回logits1预测值, kl_loss损失
        logits, _ = model(input_ids=input_ids,
            token_type_ids=token_type_ids,
            do_evaluate=True)
        # model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True)
        loss = criterion(logits, labels)#交叉熵损失
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)#acc
        metric.update(correct)#更新metric
        accu = metric.accumulate()

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    logging.info("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu


def pre_dev():
    # 设置设备
    paddle.set_device(args.device)
    
    # 多卡训练时当前进程的rank
    rank = paddle.distributed.get_rank()
    
    #动态图
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        
    # 设置种子
    set_seed(args.seed)
    # 加载验证集
    dev_ds = load_dataset(
        read_text_pair, data_path=args.dev_set, is_test=False, lazy=False)
    
    # 选择模型
    pretrained_model = ErnieGramModel.from_pretrained('ernie-gram-zh')
    tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
        
  

                                                                                     
    #包装转换函数，返回input_ids（分词）, token_type_ids（分句）
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        #填充token
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input1
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment2
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    # 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
    # 长度是指的： 一个batch中的最大长度，主要考虑性能开销
    # paddlenlp.data.Tuple	将多个batchify函数包装在一起

    #创造dataloader，返回paddle.io.DataLoader类型
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    #构建模型
    model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef)

    #设置模型路径
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print('load model:',args.init_from_ckpt)
    
    #通过数据并行模式执行动态图模型
    #使用 paddle.distributed.launch 方法启动
    model = paddle.DataParallel(model)
   
    #计算输入input和标签label间的交叉熵损失 
    criterion = paddle.nn.loss.CrossEntropyLoss()
    
    #评估
    metric = paddle.metric.Accuracy()

    accuracy = evaluate(model, criterion, metric, dev_data_loader)
            
if __name__ == "__main__":
    pre_dev()         