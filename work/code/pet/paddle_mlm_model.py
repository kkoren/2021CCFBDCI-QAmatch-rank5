import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp

class ErnieGramLMPredictionHead(nn.Layer):#计算mask概率
    """
    Ernie Gram Model with a `language modeling` head on top.
    """

    def __init__(
            self,
            hidden_size,#隐藏层大小
            vocab_size,#词汇表大小
            activation,#激活函数
            embedding_weights=None,
            weight_attr=None, ):
        super(ErnieGramLMPredictionHead, self).__init__()

        
        self.transform = nn.Linear(hidden_size, hidden_size, weight_attr=weight_attr)#Out=XW+b
        self.activation = getattr(nn.functional, activation)#getattr() 函数用于返回一个对象属性值，获得原来的激活函数
        self.layer_norm = nn.LayerNorm(hidden_size)#归一化层

        #权重
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            attr=weight_attr,
            is_bias=False) if embedding_weights is None else embedding_weights

        #偏重
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:#[Mask]位置不为空

            #改变隐藏层状态,改为二维，第二个维度为隐藏层最后的维度
            #-1 表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
            #0 表示实际的维数是从x的对应维数中复制出来的
            
            hidden_states = paddle.reshape(hidden_states,[-1, hidden_states.shape[-1]])

            #根据索引 index （mask位置）获取输入 input 指定条目，并将它们拼接在一起。
            hidden_states = paddle.tensor.gather(hidden_states,masked_positions)
        
        # gather masked tokens might be more quick
        #线性层，全连接 shape [768,768]
        hidden_states = self.transform(hidden_states)
        #激活层
        hidden_states = self.activation(hidden_states)
        #层归一化层,对隐藏层做层归一化（对某一层所有神经元的输入做归一化）
        hidden_states = self.layer_norm(hidden_states)

        #decoder层，计算两个Tensor的乘积，遵循完整的广播规则,[seq_len,768]*[768,vocab_size]=[seq_len,vocab_size]
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias

        return hidden_states

class ErnieGramPretrainingHeads(nn.Layer):
    '''
     - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

    - `seq_relationship_score` (Tensor):
        The scores of next sentence prediction.
        Its data type should be float32 and its shape is [batch_size, 2].
    '''
    def __init__(
            self,
            hidden_size,
            max_position_embeddings,
            vocab_size,
            activation,
            embedding_weights=None,
            weight_attr=None, ):
        super(ErnieGramPretrainingHeads, self).__init__()

        self.predictions = ErnieGramLMPredictionHead(hidden_size, vocab_size, activation, embedding_weights, weight_attr)#预测MASK的概率

        self.dropout = nn.Dropout(0.1)
        self.linear=nn.Linear(hidden_size,max_position_embeddings)#全连接层
        self.batchnorm1d=nn.BatchNorm1D(max_position_embeddings)#BN层
        self.relu=nn.Relu()#激活函数
        self.classifier = nn.Linear(max_position_embeddings, 2)#全连接层

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        #预测[MASK]的概率
        prediction_scores = self.predictions(sequence_output, masked_positions)

        #预测[CLS]两句话的上下文的概率
        cls_embedding1 = self.dropout(pooled_output)#dropout层
        cls_embedding1=self.linear(cls_embedding1)#全连接层
        cls_embedding1=self.batchnorm1d(cls_embedding1)#BN层
        cls_embedding1=self.relu(cls_embedding1)#激活层
        cls_embedding1=self.dropout(cls_embedding1)#dropout
        seq_relationship_score = self.classifier(cls_embedding1)#全连接层

        return prediction_scores, seq_relationship_score

class QuestionMatching_mlm(nn.Layer):
    '''
    base模型
    dropout改为0.15
    '''
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model

        #创建一个参数属性对象,权重是随机正态分布
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=self.ptm.initializer_range))
        
        self.cls = ErnieGramPretrainingHeads(
            self.ptm.config["hidden_size"],
            self.ptm.config["max_position_embeddings"],
            self.ptm.config["vocab_size"],
            self.ptm.config["hidden_act"],
            embedding_weights=self.ptm.embeddings.word_embeddings.weight,
            weight_attr=weight_attr, )

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False,
                masked_positions=None):

        #sequence_output最后一层输出的隐藏状态，[batch_size,seq_len,hidden_size]
        #pooled_output[CLS]最后一层的状态,[batch_size,hidden_size]
        sequence_output,pooled_output = self.ptm(input_ids, token_type_ids,position_ids,attention_mask)
        prediction_scores, seq_relationship_score=self.cls(sequence_output,pooled_output, masked_positions)
        
        
        return prediction_scores, seq_relationship_score