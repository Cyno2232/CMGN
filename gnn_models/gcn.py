import dgl.function as fn
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Dropout
import torch.nn.functional as F
from typing import List
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn.activations import Activation

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce_sum = fn.sum(msg='m', out='h')
gcn_reduce_mean = fn.mean(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, init_range):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.to_init(init_range)
        self.activation = activation

    def to_init(self, init_range):
        # print("Linear")
        for param in self.linear.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)
            elif len(param.shape) == 1:
                init.zeros_(param)

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, init_range):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation, init_range)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce_mean)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

# def hook_fn(module, input, output):
#     # 输出的张量是 output，可以在此处对其进行打印、记录或分析操作
#     print("Output shape:", output.shape)
#     print("Output values:")
#     print(output)
#
# def hook_fn_backward(module, grad_input, grad_output):
#     # 梯度的张量是 grad_output[0]，可以在此处对其进行打印、记录或分析操作
#     print("Gradient shape:", grad_output[0].shape)
#     print("Gradient values:")
#     print(grad_output[0])

class GCNNet(nn.Module):
    def __init__(self, hdim, init_range, nlayers: int = 2, dropout_prob: int = 0.1):
        super(GCNNet, self).__init__()
        self._gcn_layers = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []
        feedfoward_input_dim, feedforward_hidden_dim, hidden_dim = hdim, hdim, hdim
        for i in range(nlayers):
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_output_dim())
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            gcn = GCN(hdim, hdim, F.relu, init_range)
            self.add_module(f"gcn_{i}", gcn)
            self._gcn_layers.append(gcn)

            layer_norm = LayerNorm(hdim)
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = Dropout(dropout_prob)
        self._input_dim = hdim
        self._output_dim = hdim

    def forward(self, g, features):
        output = features

        for i in range(len(self._gcn_layers)):
            gcn = getattr(self, f"gcn_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")

            cached_input = output
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # handle1 = gcn.register_forward_hook(hook_fn)
            # handle2 = gcn.register_backward_hook(hook_fn_backward)
            attention_output = gcn(g, feedforward_output)
            # handle1.remove()
            # handle2.remove()
            output = layer_norm(self.dropout(attention_output) + feedforward_output)

        return output
