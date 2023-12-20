import pickle

with open('word_embedding.p', 'rb') as f:
    temp = pickle.load(f)
    print(len(temp[0]))