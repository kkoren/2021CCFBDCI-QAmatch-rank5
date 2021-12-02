# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from paddle_mlm_data import*
from paddle_mlm_model import *

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--result_file", type=str, required=True, help="The result file name")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=62, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.
    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    
    with paddle.no_grad():
        for batch_data in data_loader:
   
            input_ids,token_type_ids,target_ids = batch_data

            input_ids= paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            #预测batch
            batch_prediction_scores,batch_seq_relationship_score, kl_loss= model(
                input_ids=input_ids, token_type_ids=token_type_ids,
                )
            
            hen_logits=batch_prediction_scores[:,1,pos_id]
            hen_logits=paddle.reshape(hen_logits,[-1,1])
            bu_logits=batch_prediction_scores[:,1,neg_id]
            bu_logits=paddle.reshape(bu_logits,[-1,1])

            batch_logit=paddle.concat([bu_logits,hen_logits],axis=-1)
            
            batch_logits.append(batch_logit.numpy())
        #总预测
        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits

def save(test_ds,y_preds,ifold):
    test_ds['label']=y_preds
    test_ds['label']=test_ds['label'].astype(int)
    test_ds['label'].to_csv(args.result_file+'_'+ifold+'.csv',header=None, index=False)
    
if __name__ == "__main__":
    paddle.set_device(args.device)

    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

    trans_func = partial(
        convert_query_to_ids,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id), # target_ids
        
    ): [data for data in fn(samples)]


    # 对应的任务描述
    prefix = u'很相似句子对：'
    mask_idx=1 #掩盖pattern中字的位置,因为还有cls所以为1
    global pos_id,neg_id
    pos_id = tokenizer(u'很')['input_ids'][1]#取很的id
    neg_id = tokenizer(u'不')['input_ids'][1]#取不的id


    test_ds = load_dataset(
        read_text_pair, data_path=args.input_file, is_test=True, lazy=False)

    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)


    

    
    model = QuestionMatching_mlm(pretrained_model)
        
      
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")
        
    y_probs = predict(model, test_data_loader)
    #取概率大的
    y_preds = np.argmax(y_probs, axis=1)

    test_ds=read_csvToDF(data_path=args.input_file, is_test=True)

    save(test_ds,y_preds,'pet')


            
   