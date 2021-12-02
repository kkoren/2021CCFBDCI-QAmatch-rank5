"""
本文件定义GNN网络
"""

from paddle.fluid import layers
from ddparser.parser.nets import nn
import paddle

class GraphAttentionLayer(paddle.nn.Layer):
    """GraphAttentionLayer"""
    def __init__(self,input_size, output_size,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()

        #全连接
        self.fc=paddle.nn.Linear(input_size,output_size)
        #激活函数
        self.leakyrelu=paddle.nn.LeakyReLU(alpha)
        self.dropout1 = paddle.nn.Dropout(dropout if dropout is not None else 0.15)
        
        self.concat = concat#是否连接
        #创建参数Q
        self.a1 = paddle.create_parameter(shape=[output_size, 1], dtype='float32')
        self.a2 = paddle.create_parameter(shape=[output_size, 1], dtype='float32')
        
       
    def forward(self, squence_out, adj):
        """Forward network"""
        #x[128,68,768]
        #adj[128,68,68]

        #线性变换  [128,68,768]=>[128,68,100]
        squence_out = self.fc(squence_out)#全连接

        _, N, _ = squence_out.shape

        #点积 squence_out=K=V[128,68,100]*Q[100,1]=>[128,68,1]=>[128,68,68]
        middle_result1 = paddle.tile(paddle.matmul(squence_out, self.a1), repeat_times=[1, 1, N])

        #点积 squence_out=K=V[128,68,100]*Q[100,1]=>[128,68,1]=>[128,68,68]=>[128,68,68]
        middle_result2 = paddle.transpose(paddle.tile(paddle.matmul(squence_out, self.a2), repeat_times=[1, 1, N]),perm=[0, 2, 1])

        #LeakyReLU 激活层 [128,68,68]
        e = self.leakyrelu(middle_result1 + middle_result2)

        #变类型
        adj = paddle.cast(adj, dtype='int32')

        #[128,68,68]
        attention = nn.mask_fill(e, adj == 0.0, -1e9)
        attention = paddle.nn.functional.softmax(attention, axis=2)
        attention =self.dropout1(attention)

        #点积 [128,68,68]*[128,68,100] =>[128,68,100]
        h_prime = paddle.matmul(attention, squence_out)
        
        if self.concat:
            return paddle.nn.functional.elu(h_prime)
        else:
            return h_prime


class GAT(paddle.nn.Layer):
    """GAT"""
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, heads, layer):
        super().__init__()
        self.dropout = paddle.nn.Dropout(dropout if dropout is not None else 0.15)
    
        self.layer = layer#层数
        if self.layer == 1:
            self.attentions =  paddle.nn.LayerList([
                GraphAttentionLayer(input_size, output_size, dropout=dropout, alpha=alpha, concat=True)
                for _ in range(heads)])
        else:
            self.attentions = paddle.nn.LayerList(
                [GraphAttentionLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in range(heads)])
            
            self.out_att = GraphAttentionLayer(hidden_size * heads,
                                               output_size,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)

    def forward(self, squence_out, adj):
        """Forward network"""

       
        squence_out = self.dropout(squence_out)

        if self.layer == 1:
            gat_emb= paddle.stack([att(squence_out, adj) for att in self.attentions], dim=2)
            gat_emb = paddle.sum(gat_emb, 2)
            gat_emb = self.dropout(gat_emb)
            return paddle.nn.functional.log_softmax(gat_emb, axis=2)
        else:
            #12头多头注意力机制
            gat_emb=[att(squence_out, adj) for att in self.attentions]
            #拼接
            gat_emb = paddle.concat(gat_emb, axis=2)
            gat_emb=self.dropout(gat_emb)
            gat_emb=self.out_att(gat_emb,adj)
            return gat_emb