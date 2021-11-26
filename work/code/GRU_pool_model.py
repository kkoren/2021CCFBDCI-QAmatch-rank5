import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddlenlp as ppnlp


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0,gru_hidden_size=192):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.2)

        self.dropout1 = nn.Dropout(dropout if dropout is not None else 0.4)
        
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"]*2, 2)
        self.Linear=nn.Linear(self.ptm.config["hidden_size"],self.ptm.config["hidden_size"])
        
        #GRU
        self.rdrop_coef = rdrop_coef
        self.gru_layer=nn.GRU(input_size=768,hidden_size=gru_hidden_size,num_layers=1,direction='bidirect',dropout=0.1)

        #转置，压缩
        
        #maxpool参数（）
        self.max_pool=nn.MaxPool1D(kernel_size=62, stride=62, padding='SAME')
        self.avg_pool=nn.AvgPool1D(kernel_size=62, stride=62, padding='SAME')
        self.sigmoid=nn.Sigmoid()
        #self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):

        sentence_embedding, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        #sentence_embedding的size:batch_size,sequence_length,hidden_size
        #cls_embedding的size:batchsize.hidden_size
        #cls_embedding进入全连接层
        cls_out=self.Linear(cls_embedding1)
        cls_out=self.dropout(cls_out)
        #sentence-embedding进入GRU
        gru_out,(last_hidden,_)=self.gru_layer(sentence_embedding)
        #sentence_embedding进入最大池化层
        
        trans_gru_out=fluid.layers.transpose(gru_out,perm=[0,2,1])
        
        max_embedding=self.max_pool(trans_gru_out)
        #max_squ=paddle.squeeze(max_embedding)
        max_squ=paddle.mean(max_embedding,axis=2)

        #sentence_embedding进入平均池化层
        avg_embedding=self.avg_pool(trans_gru_out)
        #avg_squ=paddle.squeeze(avg_embedding)
        avg_squ=paddle.mean(avg_embedding,axis=2)

        #拼接
        all_embedding=paddle.concat([cls_out,max_squ,avg_squ],axis=1)

        #sigmoid
        all_sigmoid=self.sigmoid(all_embedding)

        dropout_out = self.dropout1(all_sigmoid)

        logits1 = self.classifier(dropout_out)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            #kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss