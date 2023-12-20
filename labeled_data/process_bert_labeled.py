# -*- coding: utf8 -*-
import pickle
import os
import nltk
import re
from tqdm import tqdm

def handle_data(word_to_id, data, hmt_max_sentence_length, hmt_max_content_length):
    new_data = []
    for each in tqdm(data, total=len(data)):
        id_ = each[0]
        label = each[5]
        contents_token = each[6]
        abstracts_token = each[7]

        cur_max = 0
        for s in contents_token:
            if len(s) > cur_max:
                cur_max = len(s)

        if cur_max < hmt_max_sentence_length  and len(contents_token) <= hmt_max_content_length:

            contents_id = []
            ########## handle ids #######
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
                contents_padding.append([0]*hmt_max_sentence_length)
                contents_mask.append([0]*hmt_max_sentence_length)

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

def get_max_length(data):
    max_sentence_length = 0
    max_content_length = 0
    for each in data:
        id_, _, _, _, _, _, contents_token, abstracts_token = each
        for s in contents_token:
            if len(s) > max_sentence_length:
                max_sentence_length = len(s)

        cur_content_length = len(contents_token)
        if cur_content_length > max_content_length:
            max_content_length = cur_content_length

    return max_sentence_length, max_content_length

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

def get_info(final_tokens, id_):
    find = False
    find_index = -1
    for i in range(len(final_tokens)):
        cur_id, _, _, _, _ = final_tokens[i]
        if cur_id == id_:
            find = True
            find_index = i
            break

    if find:
        cur = final_tokens[find_index]
        return cur[1], cur[2], cur[3], cur[4]
    else:
        print(id_)

def tokenize_data(data, final_tokens):
    new_data = []
    for each in data:
        id_, _, _, label = each
        contents, abstracts, contents_for_token, abstracts_for_token = get_info(final_tokens, id_)
        contents_token, abstracts_token = [], []

        for s in contents_for_token:
            c_tokens = nltk.word_tokenize(clean_str(s))
            contents_token.append(c_tokens)

        for s in abstracts_for_token:
            a_tokens = nltk.word_tokenize(clean_str(s))
            abstracts_token.append(a_tokens)

        new_data.append([id_,
                         contents,
                         abstracts,
                         contents_for_token,
                         abstracts_for_token,
                         label,
                         contents_token,
                         abstracts_token
        ])
    return new_data

def main():
    final_token_files = "../../generate_training_data/final_dataset/final_token.p"
    data_paths = ["train_labeled_0_5000.p"]
    valid_data = "valid_labeled.p"
    writing_path = "train_data_0_5000/"
    os_exists = os.path.exists(writing_path)

    if not os_exists:
        os.mkdir(writing_path)

    final_tokens, final_token_dev = pickle.load(open(final_token_files, "rb"))
    train_data = []
    for e in data_paths:
        t_d = pickle.load(open(e, "rb"))[0]
        train_data.extend(t_d)
    dev_data = pickle.load(open(valid_data, "rb"))[0]

    train_data_token = tokenize_data(train_data, final_tokens)
    dev_data_token = tokenize_data(dev_data, final_token_dev)

    max_sentence_1, max_list_1 = get_max_length(train_data_token)
    max_sentence_2, max_list_2 = get_max_length(dev_data_token)

    max_sentence_length = max([max_sentence_1, max_sentence_2])
    max_content_length = max([max_list_1, max_list_2])

    print("max sentence length: ", max_sentence_length)
    print("max content length: ", max_content_length)

    ######### build vocabulary ###########
    word_to_index, index_to_word, _ = pickle.load(open("../../labeling/model/my_vocabulary_add_padd.pickle", "rb"))

    hmt_max_sentence_length = 50
    hmt_max_content_length = 50

    ####### handle data
    train_id = handle_data(word_to_index, train_data_token, hmt_max_sentence_length, hmt_max_content_length)
    valid_id = handle_data(word_to_index, dev_data_token, hmt_max_sentence_length, hmt_max_content_length)
    print("train ", len(train_data_token), len(train_id))
    print("dev ", len(dev_data_token), len(valid_id))

    pickle.dump([train_id, valid_id], open(writing_path + "/data_info.p", "wb"))
    pickle.dump([max_sentence_length, max_content_length, hmt_max_sentence_length, hmt_max_content_length], open(writing_path + "/max_info.p", "wb"))
    ### word_id_info ../model/my_vocabulary_add_padd.pickle
    ### embedding ../word_embedding 还没有<pad>

if __name__ == '__main__':
    main()
