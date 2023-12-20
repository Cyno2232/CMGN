import time

import torch
import torch.nn as nn
from torch import optim
from general_utils import get_logger
import pickle
import numpy as np
import random
from torch.backends import cudnn
from model_rl_gcn import Seq2Graph_rl_gcn
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
from tqdm import tqdm
import math

from api import evaluation_sim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Process some hyper parameters")

# train的batch size是64   test的时候是32【预测testing的120的时候】

# training
parser.add_argument("--optimizer", dest="optimizer", default="adam")
parser.add_argument("--lr", dest="lr", default=1e-4, type=float)
parser.add_argument("--gcl_lr", dest="gcn_lr", default=1e-4, type=float)
parser.add_argument("--gcn_layer_num", dest="gcn_layer_num", default=2, type=int)
parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("--test_batch_size", dest="test_batch_size", default=64, type=int)
parser.add_argument("--nepochs", dest="nepochs", default=120)
parser.add_argument("--data_path", dest="data_path", default="labeled_data/train_data_0_5000/")
parser.add_argument("--start", dest="start", default=0)
parser.add_argument("--end", dest="end", default=5000)
parser.add_argument("--emb_path", dest="emb_path", default="labeled_data/embedding/")
parser.add_argument("--nepoch_no_improv", dest="nepoch_no_imprv", default=3, type=int)
parser.add_argument("--device", dest="device", default=0, type=int)
parser.add_argument("--model_type", dest="model_type", default="gcn")
parser.add_argument("--model_name", dest="model_name", default="20230809_test", type=str)

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
parser.add_argument("--gcl_eta", dest="gcl_eta", type=float, default=0.3, help='0.1, 1.0, 10, 100, 1000')
parser.add_argument("--remove_edges", dest="remove_edges", type=int, default=0)
parser.add_argument("--remove_nodes", dest="remove_nodes", type=int, default=0)
parser.add_argument("--label_threshold", dest="label_threshold", default=0.6, type=float)

parser.add_argument("--clamp_min", dest="clamp_min", default=0.05, type=float)

parser.add_argument("--seed", dest="seed", default=2, type=int)
parser.add_argument("--register", dest="register", default=True, type=bool)

config = parser.parse_args()

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

def get_performance(crit, pred, labels):
    loss = crit(pred, labels)
    total_loss = loss

    pred = pred.max(1)[1]
    n_correct = pred.data.eq(labels.data)
    n_correct = n_correct.sum()

    return total_loss, n_correct, pred

def adjust_learning_rate_my(optimizer, p):
    lr = 0.0075 / math.pow(1+10*p, 0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("adjust learning rate to: " + str(lr))

def get_maps(maps, device, step, batch_size):
    if (step + 1) * batch_size > len(maps):
        map_batch = maps[step * batch_size: len(maps)]
    else:
        map_batch = maps[step * batch_size: (step + 1) * batch_size]

    if isinstance(map_batch[0], np.ndarray):
        return map_batch

    maps_cuda = []
    for m in map_batch:
        # graph_visualization(m)
        # m = dgl.reverse(m)
        # graph_visualization(m)
        cuda_m = m.to(device)
        maps_cuda.append(cuda_m)

    return maps_cuda


def random_remove_edge(maps, num=0):
    """
    num: the number of the removed edges
    """
    modified_maps = []

    if num >= 0:  # Remove edges
        for each in maps:
            indices = np.random.choice(each.number_of_edges(), num, replace=False)
            modified_graph = each.clone()
            modified_graph.remove_edges(indices)
            modified_maps.append(modified_graph)
    else:  # Add edges
        num_to_add = abs(num)
        for each in maps:
            modified_graph = each.clone()
            num_nodes = modified_graph.number_of_nodes()

            for i in range(num_to_add):
                while True:
                    src = np.random.randint(0, num_nodes)
                    dst = np.random.randint(0, num_nodes)
                    if not modified_graph.has_edges_between(src, dst):
                        break
                modified_graph.add_edges(src, dst)

            modified_maps.append(modified_graph)

    return modified_maps


def train_epoch(seq2graph_model, data_loader, length, crit, optimizer, config):
    seq2graph_model.train()

    total_loss = 0
    if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
        # with open('maps/maps_final.p', 'rb') as f:
        with open('directed_maps/DGLgraph/train.p', 'rb') as f:
            total_maps = pickle.load(f)

    if config.remove_edges != 0:
        modified_maps = random_remove_edge(total_maps, config.remove_edges)
    else:
        modified_maps = total_maps

    for step, batch in enumerate(tqdm(data_loader, total=length)):
        # if step < 150:
        #     continue
        if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
            maps = get_maps(modified_maps, device, step, config.batch_size)
            loss = seq2graph_model(batch, crit, maps)
        else:
            loss = seq2graph_model(batch, crit, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def eval_epoch(seq2graph_model, data_loader, length, crit, config):
    seq2graph_model.eval()

    total_loss = 0
    if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
        # with open('maps/maps_final_valid.p', 'rb') as f:
        with open('directed_maps/DGLgraph/valid.p', 'rb') as f:
            total_maps = pickle.load(f)

    for step, batch in enumerate(tqdm(data_loader, total=length)):
        if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
            maps = get_maps(total_maps, device, step, config.batch_size)
            loss, adj = seq2graph_model.inference(batch, crit, maps)
        else:
            loss, adj = seq2graph_model.inference(batch, crit)
        # forward
        # loss, adj = seq2graph_model.inference(batch, crit)
        total_loss += loss.item()
    return total_loss

def change_test_data(test_data, id_, cur_adj):
    for i in range(len(test_data)):
        cur_id = test_data[i][0]
        if cur_id == id_:
            test_data[i].append(cur_adj)
            break
    return test_data

def eval_test_epoch(seq2graph_model, test_data, data_loader, length, crit, config):
    """Epoch operation in evaluation phase"""
    seq2graph_model.eval()

    total_loss = 0
    total_results = []
    if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
        # with open('maps/maps_final_dev.p', 'rb') as f:
        with open('directed_maps/DGLgraph/dev.p', 'rb') as f:
            total_maps = pickle.load(f)

    for step, batch in enumerate(tqdm(data_loader, total=length)):
        if isinstance(seq2graph_model, Seq2Graph_rl_gcn):
            maps = get_maps(total_maps, device, step, config.batch_size)
            loss, adjs = seq2graph_model.inference(batch, crit, maps)
        else:
            loss, adjs = seq2graph_model.inference(batch, crit)
        # forward
        # loss, adjs = model.inference(batch, crit)
        for i in range(len(batch[0])):
            cur_id = batch[0][i]
            cur_adjs = adjs[i]
            test_data = change_test_data(test_data, cur_id, cur_adjs)

    return test_data

def eval_score(seq2graph_model, crit, config):
    # path = "../testing_final_20200909/processed_for_seq2graph_dev/dev_for_mymodel_info.p"
    # full_dev_data = "../testing_final_20200909/processed/dev_full.p"
    # max_info = "../testing_final_20200909/processed_for_seq2graph_dev/max_info.p"
    # old code above
    path = "testing_final_20201012/processed_for_seq2graph_dev/dev_for_mymodel_info.p"
    full_dev_data = "testing_final_20201012/processed/dev_full.p"
    max_info = "testing_final_20201012/processed_for_seq2graph_dev/max_info.p"

    dev_data = pickle.load(open(path, "rb"))[0]
    max_sentence_length, max_content_length = pickle.load(open(max_info, "rb"))

    full_dev_data = pickle.load(open(full_dev_data, "rb"))[0]

    seq2graph_model.max_sentence_length = max_sentence_length
    seq2graph_model.max_content_length = max_content_length

    dev_data_loader = data_loader(dev_data, config.batch_size)
    length = math.floor(len(dev_data) / config.batch_size) + 1
    print("deving evaluation.......")
    total_ = eval_test_epoch(seq2graph_model, full_dev_data, dev_data_loader, length, crit, config)
    # pickle.dump([total_], open("../testing_final_20200909/to_dev/dev_" + config.model_type + ".p", "wb"))

    score, word_score = evaluation_sim(total_, "dev", config.model_type)

    return score

def train(seq2graph_model, crit, optimizer, logger, train_data, valid_data, config):
    max_score = 0
    max_test_score = 0
    no_imprv = 0

    for epoch_i in range(config.nepochs):
        train_data_loader = data_loader(train_data, config.batch_size)
        dev_data_loader = data_loader(valid_data, config.test_batch_size)

        train_length = math.floor(len(train_data) / config.batch_size) + 1
        dev_length = math.floor(len(valid_data) / config.test_batch_size) + 1

        logger.info("---------Epoch {:} out of {:}---------".format(epoch_i+1, config.nepochs))

        print("training.....")
        train_loss = train_epoch(seq2graph_model, train_data_loader, train_length, crit, optimizer, config)
        train_msg = "loss: {:04.5f}".format(train_loss)
        logger.info(train_msg)

        print("dev evaluation.......")
        start = time.time()
        dev_loss = eval_epoch(seq2graph_model, dev_data_loader, dev_length, crit, config)
        end = time.time()
        print("time ", end - start)
        dev_msg = "dev loss: {:04.6f}".format(dev_loss)
        logger.info(dev_msg)

        print("deving 15 files .......")
        dev_score = eval_score(seq2graph_model, crit, config)
        logger.info(dev_score)

        seq2graph_model.max_sentence_length = config.max_sentence_length
        seq2graph_model.max_content_length = config.max_content_length

        model_name = config.dir_output + config.model_type + "_" + config.model_name + ".pkl"
        if dev_score > max_score:
            no_imprv = 0
            torch.save(seq2graph_model.state_dict(), model_name)
            logger.info("new best score!!" + "The checkpoint file has been updated.")
            max_score = dev_score
        else:
            no_imprv += 1
            if no_imprv >= config.nepoch_no_imprv:
                break


def test_target_zero(data):
    for each in data:
        _, _, _, asp, _, domain = each
        if len(asp) == 0:
            print(asp)
            print(domain)

def expand_data(data):
    new_data = []
    for each_cate in data:
        new_data.extend(data[each_cate])

    return new_data

def get_source_first_id(data):
    l = []
    for each_domain in data:
        ins = each_domain[0]
        first_id = ins[3]
        l.append(first_id)
    return l

def main():
    config.dir_output = "results_" + str(config.seed) + "/"
    path_log = config.dir_output

    if not os.path.exists(path_log):
        os.makedirs(path_log)

    path = config.data_path + "data_info.p"
    max_info = config.data_path + "max_info.p"

    train_data, valid_data = pickle.load(open(path, "rb"))
    _, _, max_sentence_length, max_content_length = pickle.load(open(max_info, "rb"))

    config.max_sentence_length = max_sentence_length
    config.max_content_length = max_content_length

    logger = get_logger(path_log+"log.txt")
    logger.info(config)

    if config.model_type == "gcn":
        seq2graph_model = Seq2Graph_rl_gcn(config, max_sentence_length, max_content_length)
    else:
        print("Error! No model type")
        return

    crit = nn.MSELoss(reduction="mean")

    if torch.cuda.is_available():
        print("cuda availiable !!!!!!!!")
        print(f'running on gpu {torch.cuda.current_device()}!!!!!!!')
        seq2graph_model = seq2graph_model.to(device)
        crit = crit.to(device)

    num_params = 0
    for param in seq2graph_model.parameters():
        num_params += param.numel()
    print("number of parameters: ", num_params)

    para_list = filter(lambda p: p.requires_grad, seq2graph_model.parameters())
    if config.optimizer == "sgd":
        optimizer = optim.SGD(para_list, lr=config.lr, momentum=0.9)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    elif config.optimizer == "adadelta":
        optimizer = optim.Adadelta(para_list, lr=config.lr)

    train(seq2graph_model, crit, optimizer, logger, train_data, valid_data, config)

if __name__ == '__main__':
    main()