import numpy as np

__all__ = ['Pad_batch_graphs']

class Pad_batch_graph(object):#根据句子最长长度填0
    def __init__(self,axis=None):
        self._axis = axis
    def __call__(self, graphs):
        max_len = max([s.shape[self._axis] for s in graphs])

        out_dims = (len(graphs),max_len ,max_len )#维度
        out_tensor = np.full(out_dims, 0, dtype=np.int64)#填充一个空矩阵

        for i, tensor in enumerate(graphs):
            length = min(tensor.shape[0], max_len)
            out_tensor[i, :length, :length] = tensor
            
        return out_tensor