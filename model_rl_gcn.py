import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os
import pickle
import time
from sklearn.cluster import KMeans
from collections import defaultdict

from general_utils import rouge_sim2
from gnn_models.gcn import GCNNet
from gnn_models.gcl import GCL_model


class Seq2Graph_rl_gcn(nn.Module):
    def __init__(self, config, max_sentence_length, max_content_length):
        super(Seq2Graph_rl_gcn, self).__init__()

        emb_path = config.emb_path + "word_embedding.p"
        if os.path.exists(emb_path):
            word_embeddings = pickle.load(open(emb_path, "rb"), encoding="iso-8859-1")[0]
        else:
            print("Error: Can not find embedding")

        vocab_size = len(word_embeddings)
        self.word_embeds = nn.Embedding(vocab_size, config.dim_word, padding_idx=0)
        pretrained_weight_word = np.array(word_embeddings)
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight_word))
        self.word_embeds.weight.requires_grad = True

        self.config = config
        self.device = torch.device("cuda:"+str(config.device) if torch.cuda.is_available() else "cpu")
        self.max_sentence_length = max_sentence_length
        self.max_content_length = max_content_length

        self.lstm_for_sentence = nn.LSTM(
            input_size=config.dim_word,
            hidden_size=config.lstm_hidden,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_for_graph = nn.LSTM(
            input_size=config.dim_word,
            hidden_size=config.lstm_hidden,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.W_dim = config.lstm_hidden * 2

        self.U = nn.Parameter(torch.FloatTensor(self.W_dim, self.W_dim))
        self.prev = nn.Parameter(torch.FloatTensor(self.W_dim, self.W_dim))
        self.later = nn.Parameter(torch.FloatTensor(self.W_dim, self.W_dim))

        self.gcn = GCNNet(config.dim_word, config.initializer_range, nlayers=config.gcn_layer_num)
        self.gcl = GCL_model(config,
                             num_layers=5,
                             num_mlp_layers=2,
                             input_dim=config.dim_word,
                             hidden_dim=config.dim_word,
                             output_dim=config.dim_word,
                             final_dropout=0.1,
                             learn_eps=False,
                             graph_pooling_type='sum',
                             neighbor_pooling_type='sum',
                             max_content_length=self.max_content_length
                             )

        self.init_weight(config)
        print("weights initialization finished")

    def to_init(self, param, config):
        init.normal_(param, mean=0.0, std=config.initializer_range)

    def init_weight(self, config):
        print("initializing weights")
        self.to_init(self.U, config)
        self.to_init(self.prev, config)
        self.to_init(self.later, config)

        for m in self.children():
            if isinstance(m, nn.LSTM):
                print("LSTM")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.normal_(param, mean=0.0, std=config.initializer_range)
                    elif len(param.shape) == 1:
                        init.zeros_(param)
            elif isinstance(m, nn.Linear):
                print("Linear")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.normal_(param, mean=0.0, std=config.initializer_range)
                    elif len(param.shape) == 1:
                        init.zeros_(param)
            elif isinstance(m, GCNNet):
                print("GCNNet")
                # for param in m.parameters():
                #     if len(param.shape) >= 2:
                #         init.normal_(param, mean=0.0, std=config.initializer_range)
                #     elif len(param.shape) == 1:
                #         init.zeros_(param)
            elif isinstance(m, GCL_model):
                print("GCL")
                # for param in m.parameters():
                #     if len(param.shape) >= 2:
                #         init.normal_(param, mean=0.0, std=config.initializer_range)
                #     elif len(param.shape) == 1:
                #         init.zeros_(param)
            elif isinstance(m, nn.Embedding):
                print("Embedding")
            elif isinstance(m, nn.Sequential):
                print("Sequential, do not initialize")
            elif isinstance(m, nn.ReLU):
                print("ReLU, do not initialize")
            else:
                print(m, type(m))
                print("Danger!!! some parameters do not initialize")

    def sentence_encoder(self, contents, content_mask):
        input = self.word_embeds(contents)

        output, _ = self.lstm_for_sentence(input)
        output = output * content_mask.unsqueeze(-1)

        out = output.max(1)[0]
        out = torch.reshape(out, [-1, self.max_content_length, self.W_dim])

        return out

    def maximum_ranking(self, prob_matrix):
        prob_vector = prob_matrix.sum(dim=1)
        indexes = prob_vector.argsort()
        indexes = list(reversed(indexes))

        return indexes, prob_vector

    def select_baseline(self, slice_adj, original_sentences):
        prob_vector = slice_adj.sum(dim=1)

        # compute baseline
        indexes = prob_vector.argsort()
        indexes = list(reversed(indexes))
        root = indexes[0]
        selected_sents = [original_sentences[root]]
        index_of_sents = list(range(slice_adj.size()[0]))
        index_of_sents.remove(root)

        feature_matrix = slice_adj.transpose(0,1)[np.ix_(index_of_sents)]

        kmeans = KMeans(n_clusters=2)
        if feature_matrix.size()[0] >= 2:
            kmeans.fit(feature_matrix.cpu().detach().numpy())
            label = kmeans.labels_

            sub_index_of_sents = defaultdict(list)
            for k, v in enumerate(label):
                sub_index_of_sents[v].append(index_of_sents[k])

            for k in sub_index_of_sents.keys():
                sub_feature_matrix = slice_adj.transpose(0,1)[np.ix_(sub_index_of_sents[k])]
                indexes, generating_probs = self.maximum_ranking(sub_feature_matrix.transpose(0,1))
                selected_sents.append(original_sentences[indexes[0]])

        return " ".join(selected_sents)

    def select_random(self, slice_adj, original_sentences):
        prob_list = []
        prob_vector = torch.softmax(slice_adj.sum(dim=1), dim=0)

        random_action = prob_vector.multinomial(1).item()
        selected_sents = [original_sentences[random_action]]
        index_of_sents = list(range(slice_adj.size()[0]))
        index_of_sents.remove(random_action)

        prob_list.append(prob_vector[random_action])

        feature_matrix = slice_adj.transpose(0, 1)[np.ix_(index_of_sents)]

        kmeans = KMeans(n_clusters=2)
        if feature_matrix.size()[0] >= 2:
            kmeans.fit(feature_matrix.cpu().detach().numpy())
            label = kmeans.labels_

            sub_index_of_sents = defaultdict(list)
            for k, v in enumerate(label):
                sub_index_of_sents[v].append(index_of_sents[k])


            for k in sub_index_of_sents.keys():
                sub_feature_matrix = slice_adj.transpose(0,1)[np.ix_(sub_index_of_sents[k])]
                indexes, generating_probs = self.maximum_ranking(sub_feature_matrix.transpose(0,1))
                generating_probs = torch.softmax(generating_probs, dim=0)
                s = generating_probs.multinomial(1).item()
                selected_sents.append(original_sentences[s])
                prob_list.append(generating_probs[s])

        return " ".join(selected_sents), prob_list

    def compute_rl_loss(self, slice_adj, original_sentences, original_abstracts):
        abstract = " ".join(original_abstracts)

        dim = slice_adj.size()[0]
        mask = 1 - torch.eye(dim, dim).to(self.device)
        slice_adj = slice_adj * mask + (1 - mask)

        # compute baseline
        baseline_select = self.select_baseline(slice_adj, original_sentences)
        baseline = rouge_sim2(abstract, baseline_select)

        # sample an action
        random_select, probs = self.select_random(slice_adj, original_sentences)
        reward = rouge_sim2(abstract, random_select)

        probs = torch.log(torch.stack(probs, dim=0))
        rl_loss = - (reward - baseline) * probs.sum()

        return rl_loss

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

    def gcn_learning(self, tensors, maps, dim, vice_model=False):
        text_tensors = torch.split(tensors, 1, dim=0)
        all_gcn_tensors = []

        for index, tensor in enumerate(text_tensors):
            num_nodes = maps[index].number_of_nodes()
            sentence_tensors = tensor.squeeze(0)[0:num_nodes, 0:dim]
            if vice_model:
                gcn_last_h = self.vice_gcn(maps[index], sentence_tensors)
            else:
                gcn_last_h = self.gcn(maps[index], sentence_tensors)
            all_gcn_tensors.append(gcn_last_h)

        gcn_tensor = self.tensor_reshape(all_gcn_tensors, self.max_content_length).to(self.device)

        return gcn_tensor


    def seq2graph(self, encoding_sentences, original_sentences, original_abstracts, maps, labels, content_length, crit, seq_masks, phase):
        output, _ = self.lstm_for_graph(encoding_sentences)
        output = output * seq_masks.unsqueeze(-1)
        cur_batch_size = output.size()[0]

        if self.config.gcn == 1:
            gcn_output = self.gcn_learning(output, maps, self.config.dim_word)
        else:
            gcn_output = output
        # if self.config.gcn_layer_num == 2:
        #     gcn_output = self.gcn_learning(gcn_output, maps, self.config.dim_word)

        prev_W = self.prev.repeat([cur_batch_size, 1, 1])
        later_W = self.later.repeat([cur_batch_size, 1, 1])
        U_W = self.U.repeat([cur_batch_size, 1, 1])

        output_prev = torch.bmm(gcn_output, prev_W)
        output_later = torch.bmm(gcn_output, later_W)
        adj = torch.bmm(output_prev, U_W)
        adj = torch.bmm(adj, torch.transpose(output_later, 1, 2))
        adj_list = torch.split(adj, split_size_or_sections=1, dim=0)

        total_loss = []
        total_adjs = []
        for i in range(len(content_length)):
            cur_adj = torch.squeeze(adj_list[i], 0)
            cur_node_num = content_length[i]
            slice_adj = cur_adj[0:cur_node_num, 0:cur_node_num]
            slice_adj = torch.sigmoid(slice_adj)
            cur_label = torch.tensor(labels[i]).float().to(self.device)

            # cur_label = torch.ge(cur_label, self.config.label_threshold).float()

            # ignore diagonal elements
            slice_adj[torch.eye(cur_node_num).byte()].detach()
            cur_label[torch.eye(cur_node_num).byte()].detach()

            if self.config.rl == 1:
                if phase == "training":
                    loss_rl = self.compute_rl_loss(slice_adj, original_sentences[i], original_abstracts[i])
                elif phase == "testing":
                    loss_rl = 0
            else:
                loss_rl = 0


            ## add weight to zeros and ones, only previous, just for record the code, not use
            """
            mask = 1 - torch.eye(cur_node_num, cur_node_num).to(self.device)
            cur_label = cur_label * mask # + (1 - mask)
            one_num = cur_label.sum()
            zero_num = cur_node_num * (cur_node_num - 1) - one_num
            weight_zero = one_num / zero_num
            weight = (1- cur_label) * weight_zero
            final_weight = weight + cur_label
            """

            loss = crit(slice_adj, cur_label)
            combined_loss = loss + self.config.rl_lambda * loss_rl

            total_loss.append(combined_loss)
            total_adjs.append(slice_adj)
        total_loss = torch.stack(total_loss)
        total_loss = torch.mean(total_loss)

        if self.config.gcl == 1:
            if phase == 'training':
                # loss_gcl = self.gcl(maps, gcn_output)
                loss_gcl = self.gcl(maps, output)
            elif phase == "testing":
                loss_gcl = 0
        else:
            loss_gcl = 0

        total_loss += self.config.gcl_lambda * loss_gcl
        return total_loss, total_adjs

    def forward(self, batch, crit, maps, phase="training"):
        ids, contents, contents_mask, _, labels, content_length, seq_masks, original_sentences, original_abstracts = batch
        encoding_sentences = self.sentence_encoder(contents, contents_mask)
        loss, _ = self.seq2graph(encoding_sentences, original_sentences, original_abstracts, maps, labels, content_length,
                                 crit, seq_masks, phase)
        return loss

    def inference(self, batch, crit, maps, phase="testing"):
        ids, contents, contents_mask, _, labels, content_length, seq_masks, original_sentences, original_abstracts = batch
        encoding_sentences = self.sentence_encoder(contents, contents_mask)
        loss, adj = self.seq2graph(encoding_sentences, original_sentences, original_abstracts, maps, labels, content_length,
                                   crit, seq_masks, phase)

        return loss, adj



