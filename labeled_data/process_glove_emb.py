import pickle

word_to_index, index_to_word, _ = pickle.load(open("../../labeling/model/my_vocabulary_add_padd.pickle", "rb"))

def init_word_embeddings(word2idx, pretrain_file, dim_word, init_std):
    import numpy as np
    wt = np.random.uniform(-init_std, init_std, [len(word2idx), dim_word])
    wt[0] = np.zeros(dim_word)
    # coding: utf-8
    find = 0
    with open(pretrain_file, encoding="utf-8") as f:
        for line in f:
            content = line.strip().split()
            if content[0] in word2idx:
                find += 1
                wt[word2idx[content[0]]] = np.array(list(map(float, content[1:])))
    return wt

dim_word = 50
init_std = 0.1

pretrin_file = "../../../../glove/glove.6B." + str(dim_word) + "d.txt"
print("loading pre-trained word vectors....")
word_embedding = init_word_embeddings(word_to_index, pretrin_file, dim_word, init_std)

pickle.dump([word_to_index, index_to_word], open("embedding/id_info.p", "wb"))
pickle.dump([word_embedding], open("embedding/word_embedding.p", "wb"))