import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch

def test_np():
    index_of_sents = list(range(10))
    print(index_of_sents)
    index_of_sents.remove(1)
    print(index_of_sents)
    print(np.ix_(index_of_sents))

def test_clustering():
    feature_matrix = torch.FloatTensor(10, 3)
    ward = AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=None)
    ward.fit(feature_matrix)
    label = ward.labels_

def test():
    l = list(np.arange(5))
    print(l)
    print(l[::-1])
    print(l[:-1][::-1])

def test_diag():
    prob_matrix = np.random.rand(5, 5)
    print(prob_matrix)
    new = np.diagonal(prob_matrix, 1)
    print(new)

    feature_matrix = prob_matrix.T[np.ix_([0, 2])]
    new2 = np.diagonal(feature_matrix.T, 1)
    print(new2)



if __name__ == '__main__':


    #feature_matrix = prob_matrix.T[np.ix_(index_of_sents)]
    #test_clustering()
    #test()
    test_diag()