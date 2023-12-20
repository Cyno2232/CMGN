# -*- coding: utf8 -*-
# from collections import defaultdict
import pickle
import os
import numpy as np
import collections
import nltk
import re
from tqdm import tqdm

def get_label(id_, bert_labels):
    label = []
    for each in bert_labels:
        cur_id_ = each[0]
        if id_ == cur_id_:
            label = each[-1]
            # print(type(label))
            break
    return label

def handle_data(word_to_id, data, hmt_max_sentence_length, hmt_max_content_length, bert_labels):
    new_data = []
    for each in tqdm(data, total=len(data)):
        id_ = each[0]
        label = get_label(id_, bert_labels)
        contents_token = each[5]
        abstracts_token = each[6]

        cur_max = 0
        for s in contents_token:
            if len(s) > cur_max:
                cur_max = len(s)

        if cur_max <= hmt_max_sentence_length and len(contents_token) <= hmt_max_content_length:
            contents_id = []
            ########### handle ids ############
            for s in contents_token:
                cur_s_id = []
                for each_w in s:
                    if each_w in word_to_id:
                        cur_s_id.append(word_to_id[each_w])
                    else:
                        cur_s_id.append(word_to_id['<unk>'])
                contents_id.append(cur_s_id)
            contents_id_length = len(contents_id)

            abstracts_id = []
            for s in abstracts_token:
                cur_s_id = []
                for each_w in s:
                    if each_w in word_to_id:
                        cur_s_id.append(word_to_id[each_w])
                    else:
                        cur_s_id.append(word_to_id['<unk>'])
                abstracts_id.append(cur_s_id)

            contents_padding = []
            contents_mask = []
            ######## padding part ##########
            for s in contents_id:
                sentence_padding = [0] * (hmt_max_sentence_length - len(s))
                sentence_mask = [1] * len(s)

                s += sentence_padding
                sentence_mask += sentence_padding

                contents_padding.append(s)
                contents_mask.append(sentence_mask)

            seq_mask = [1] * contents_id_length + \
                       [0] * (hmt_max_content_length - contents_id_length)
            for index in range(hmt_max_content_length - contents_id_length):
                contents_padding.append([0] * hmt_max_sentence_length)
                contents_mask.append([0] * hmt_max_sentence_length)

            ###### saving data ########
            cur_instance = []
            cur_instance.append(id_)
            cur_instance.append(contents_padding)
            cur_instance.append(contents_mask)

            cur_instance.append(abstracts_id)
            cur_instance.append(label)
            cur_instance.append(contents_id_length)
            cur_instance.append(seq_mask)
            cur_instance.append(each[3])
            cur_instance.append(each[4])
            new_data.append(cur_instance)
    return new_data

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_max_length(data):
    max_sentence_length = 0
    max_content_length = 0
    for each in data:
        id_, _, _, _, _, contents_token, _ = each
        for s in contents_token:
            if len(s) > max_sentence_length:
                max_sentence_length = len(s)

        cur_content_length = len(contents_token)
        if cur_content_length > max_content_length:
            max_content_length = cur_content_length

    return max_sentence_length, max_content_length

def main():
    data_path = "processed/dev_full.p"
    writing_path = "processed_for_seq2graph_dev/"
    os_exists = os.path.exists(writing_path)
    if not os_exists:
        os.mkdir(writing_path)

    labels_by_bert = "to_dev/dev_bert.p"
    bert_labels = pickle.load(open(labels_by_bert, "rb"))[0]

    data = pickle.load(open(data_path, "rb"))[0]

    max_sentence_1, max_list_1 = get_max_length(data)

    max_sentence_length = max_sentence_1
    max_content_length = max_list_1

    print("max sentence length: ", max_sentence_length)
    print("max content length: ", max_content_length)

    ####### build vocabulary #######

    word_to_index, index_to_word, _ = pickle.load(open("../labeling/model/my_vocabulary_add_padd.pickle", "rb"))

    ###### handle training data and testing data ########
    data_id = handle_data(word_to_index, data, max_sentence_length, max_content_length, bert_labels)

    # 2022-03-25 fixed bug next two lines removed 
    # pickle.dump([data_id], open(writing_path + "/dev_for_mymodel_info.p", "wb"))
    # pickle.dump([max_sentence_length, max_content_length], open(writing_path + "/max_info.p", "wb"))
    ### word_id_info ../model/my_vocabulary_add_padd.pickle
    ### embedding ../word_embedding 还没有 <pad>

if __name__ == '__main__':
    main()