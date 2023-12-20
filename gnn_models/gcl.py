import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from gnn_models.gin import GIN
import numpy as np
import pickle


class GCL_model(nn.Module):
    def __init__(self, config, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, max_content_length):
        super(GCL_model, self).__init__()

        self.config = config
        self.device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")
        self.max_content_length = max_content_length
        self.encoder = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                           output_dim, final_dropout, learn_eps, graph_pooling_type,
                           neighbor_pooling_type)
        self.vice_encoder = self.encoder
        self.proj_head = nn.Sequential(nn.Linear(self.config.dim_word, self.config.dim_word),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.config.dim_word, self.config.dim_word))



    def perturb_encoder(self, model, vice_model, config):
        for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
            std = torch.max(param.data.std(), torch.tensor(0.0))  # Ensure std >= 0.0
            # try:
            #     noise = torch.normal(0, torch.ones_like(param.data) * std).to(self.device)
            # except:
            #     noise = torch.normal(0, torch.ones_like(param.data) * torch.tensor(0.01)).to(self.device)
            noise = torch.normal(0, torch.ones_like(param.data) * std).to(self.device)
            adv_param.data = param.data + config.gcl_eta * noise
            # print(adv_param.data)

        return vice_model

    def tensor_reshape(self, tensor_list: list, sen_num):
        _, dim = tensor_list[0].size()
        tensor_pad_list = []

        for tensor in tensor_list:
            output_tensor = torch.zeros(sen_num, dim)  # 创建全0张量
            output_tensor[:tensor.shape[0], :] = tensor  # 复制张量的值
            tensor_pad_list.append(output_tensor)

        concatenated_tensor = torch.cat(tensor_pad_list, dim=0)  # 在第0个维度上拼接
        reshaped_tensor = concatenated_tensor.view(len(tensor_pad_list), sen_num, dim)  # 重塑张量的形状

        return reshaped_tensor

    def gin_learning(self, tensors, maps, dim, vice_model=False):
        text_tensors = torch.split(tensors, 1, dim=0)
        all_gcn_tensors = []

        for index, tensor in enumerate(text_tensors):
            num_nodes = maps[index].number_of_nodes()
            sentence_tensors = tensor.squeeze(0)[0:num_nodes, 0:dim]
            if vice_model:
                gcn_last_h = self.vice_encoder(maps[index], sentence_tensors)
            else:
                gcn_last_h = self.encoder(maps[index], sentence_tensors)
            all_gcn_tensors.append(gcn_last_h)

        gcn_tensor = self.tensor_reshape(all_gcn_tensors, self.max_content_length).to(self.device)

        return gcn_tensor

    def remove_zero_row(self, tensors):
        zero_rows = (tensors == 0).all(dim=1)
        nonzero_rows = tensors[~zero_rows]

        return nonzero_rows

    def projection(self, tensor, perturbed_tensor):
        batch_size, sen_num, _ = tensor.shape
        tensor = tensor.view(batch_size * sen_num, 50)
        perturbed_tensor = perturbed_tensor.view(batch_size * sen_num, 50)
        tensor = self.remove_zero_row(tensor)
        perturbed_tensor = self.remove_zero_row(perturbed_tensor)
        tensor = self.proj_head(tensor)
        perturbed_tensor = self.proj_head(perturbed_tensor)

        return tensor, perturbed_tensor

    def compute_gcl_loss(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def forward(self, g, feature):
        self.vice_encoder = self.perturb_encoder(self.encoder, self.vice_encoder, self.config)
        original_feature = self.gin_learning(feature, g, self.config.dim_word, vice_model=False)
        pertubed_feature = self.gin_learning(feature, g, self.config.dim_word, vice_model=True)
        original_proj, pertubed_proj = self.projection(original_feature, pertubed_feature)
        loss_gcl = self.compute_gcl_loss(original_proj, pertubed_proj)

        return loss_gcl

