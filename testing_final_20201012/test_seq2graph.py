import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch import optim
import os
import pickle
import numpy as np
import random
from torch.backends import cudnn
from model_rl_gcn import Seq2Graph_rl_gcn
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
import math
import time

parser = argparse.ArgumentParser(description="Process some hyper parameters")

# training
parser.add_argument("--optimizer", dest="optimizer", default="adam")
parser.add_argument("--lr", dest="lr", default=1e-4, type=float)
parser.add_argument("--gcn_lr", dest="gcn_lr", default=1e-4, type=float)
parser.add_argument("--gcn_layer_num", dest="gcn_layer_num", default=2, type=int)
parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("--test_batch_size", dest="test_batch_size", default=32, type=int)
parser.add_argument("--nepochs", dest="nepochs", default=120)
parser.add_argument("--data_path", dest="data_path", default="../labeled_data/train_data_0_5000/")
parser.add_argument("--start", dest="start", default=0)
parser.add_argument("--end", dest="end", default=5000)
parser.add_argument("--emb_path", dest="emb_path", default="../labeled_data/embedding/")
parser.add_argument("--nepoch_no_improv", dest="nepoch_no_imprv", default=5, type=int)
parser.add_argument("--device", dest="device", default=1, type=int)
parser.add_argument("--model_type", dest="model_type", default="gcn")

# training tricks
parser.add_argument("--initializer range", dest="initializer_range", default=0.02)

# model structure
parser.add_argument("--dim_word", dest="dim_word", default=50, type=int)
parser.add_argument("--lstm_hidden", dest="lstm_hidden", default=25, type=int)

parser.add_argument("--dir_output", dest="dir_output", default="results/")
parser.add_argument("--rl", dest="rl", type=int, default=1, help='1 means true, 0 means false')
parser.add_argument("--rl_lambda", dest="rl_lambda", default=0.01)

parser.add_argument("--gcn", dest="gcn", type=int, default=1, help='1 means true, 0 means false')

parser.add_argument("--gcl", dest="gcl", type=int, default=1, help='1 means true, 0 means false')
parser.add_argument("--gcl_lambda", dest="gcl_lambda", default=0.001, type=float)
parser.add_argument("--gcl_eta", dest="gcl_eta", type=float, default=0.1, help='0.1, 1.0, 10, 100, 1000')
parser.add_argument("--remove_edges", dest="remove_edges", type=int, default=0)
parser.add_argument("--remove_nodes", dest="remove_nodes", type=int, default=0)
parser.add_argument("--label_threshold", dest="label_threshold", default=0.6, type=float)

parser.add_argument("--clamp_min", dest="clamp_min", default=0.05, type=float)

parser.add_argument("--seed", dest="seed", default=2, type=int)
parser.add_argument("--model_name", dest="model_name", default='20230630_test')

config = parser.parse_args()
print(config)

seed = config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")

def data_loader(inputs, batch_size):
    inputs_size = len(inputs)
    num_blocks = int(inputs_size / batch_size) + 1
    all_ids, all_contents, all_contents_mask, all_abstracts, all_labels = [[]], [[]], [[]], [[]], [[]]
    content_length = [[]]
    all_seq_masks = [[]]
    all_sentences_content = [[]]
    all_sentences_abstract = [[]]
    block_num = 0
    for i, each in enumerate(inputs):
        if math.floor(i / batch_size) == block_num:
            all_ids[block_num].append(each[0])
            all_contents[block_num].extend(each[1])
            all_contents_mask[block_num].extend(each[2])
            all_abstracts[block_num].extend(each[3])
            all_labels[block_num].append(each[4])
            content_length[block_num].extend([each[5]])
            all_seq_masks[block_num].append(each[6])
            all_sentences_content[block_num].append(each[7])
            all_sentences_abstract[block_num].append(each[8])
        else:
            block_num += 1
            all_ids.append([])
            all_contents.append([])
            all_contents_mask.append([])
            all_abstracts.append([])
            all_labels.append([])
            content_length.append([])
            all_seq_masks.append([])
            all_sentences_content.append([])
            all_sentences_abstract.append([])

            all_ids[block_num].append(each[0])
            all_contents[block_num].extend(each[1])
            all_contents_mask[block_num].extend(each[2])
            all_abstracts[block_num].extend(each[3])
            all_labels[block_num].append(each[4])
            content_length[block_num].extend([each[5]])
            all_seq_masks[block_num].append(each[6])
            all_sentences_content[block_num].append(each[7])
            all_sentences_abstract[block_num].append(each[8])


    for i in range(num_blocks):
        yield all_ids[i], \
              torch.LongTensor(all_contents[i]).to(device), \
              torch.FloatTensor(all_contents_mask[i]).to(device), \
              all_abstracts[i], \
              all_labels[i], \
              content_length[i], \
              torch.FloatTensor(all_seq_masks[i]).to(device), \
              all_sentences_content[i], \
              all_sentences_abstract[i]

def change_test_data(test_data, id_, cur_adj):
    for i in range(len(test_data)):
        cur_id = test_data[i][0]
        if cur_id == id_:
            test_data[i].append(cur_adj)
            break
    return test_data

def eval_epoch(seq2graph_model, test_data, data_loader, length, crit, config):
    """Epoch operation in evaluation phase"""
    seq2graph_model.eval()

    total_loss = 0
    total_results = []
    if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
        # with open('../seq2graph/maps/maps_final_test.p', 'rb') as f:
        with open('../directed_maps/DGLgraph/test.p', 'rb') as f:
            total_maps = pickle.load(f)

    for step, batch in enumerate(tqdm(data_loader, total=length)):
        if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
            maps = get_maps(total_maps, device, step, config.test_batch_size)
            loss, adjs = seq2graph_model.inference(batch, crit, maps)
        else:
            loss, adjs = seq2graph_model.inference(batch, crit)
        # forward
        # loss, adjs = seq2graph_model.inference(batch, crit)
        for i in range(len(batch[0])):
            cur_id = batch[0][i]
            cur_adjs = adjs[i]
            test_data = change_test_data(test_data, cur_id, cur_adjs)

    return test_data

def train(seq2graph_model, crit, test_data, full_test_data, config):
    test_data_loader = data_loader(test_data, config.test_batch_size)
    length = math.floor(len(test_data) / config.test_batch_size) + 1
    print("deving evaluation........")
    start = time.time()
    total_ = eval_epoch(seq2graph_model, full_test_data, test_data_loader, length, crit, config)
    end = time.time()
    print("time ", end-start)
    pickle.dump([total_], open("to_test/" + str(config.seed) + "_test_" + config.model_type + f"_{config.model_name}.p", "wb"))

def main():
    path = "processed_for_seq2graph_test/test_for_mymodel_info.p"
    full_test_data = "processed/test_full.p"
    max_info = "processed_for_seq2graph_test/max_info.p"
    model_path = "../results_" + str(config.seed) + "/" + config.model_type + f"_{config.model_name}.pkl"

    test = pickle.load(open(path, "rb"))[0]
    max_sentence_length, max_content_length = pickle.load(open(max_info, "rb"))

    full_test_data = pickle.load(open(full_test_data, "rb"))[0]

    if config.model_type == "gcn":
        seq2graph_model = Seq2Graph_rl_gcn(config, max_sentence_length, max_content_length)
    else:
        print("Error! No model type")
        return

    seq2graph_model.load_state_dict(torch.load(model_path, map_location="cpu"))

    crit = nn.MSELoss(size_average=True)

    if torch.cuda.is_available():
        print("cuda availiable !!!!!!!!")
        seq2graph_model = seq2graph_model.to(device)
        crit = crit.to(device)

    train(seq2graph_model, crit, test, full_test_data, config)


def get_maps(maps, device, step, batch_size):
    map_batch = []
    if (step + 1) * batch_size > len(maps):
        map_batch = maps[step * batch_size: len(maps)]
    else:
        map_batch = maps[step * batch_size: (step + 1) * batch_size]

    if isinstance(map_batch[0], np.ndarray):
        return map_batch

    maps_cuda = []
    for m in map_batch:
        cuda_m = m.to(device)
        maps_cuda.append(cuda_m)

    return maps_cuda


if __name__ == '__main__':
    main()






