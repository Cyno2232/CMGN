import argparse
import pickle
from mind_map_generation import my_generate_mindmap_v, extractKeywords

parser = argparse.ArgumentParser(description="Process some hyper parameters")
parser.add_argument("--model_type", dest="model_type", default="self_attention")
config = parser.parse_args()
print(config)

if __name__ == '__main__':
    bert = "../testing_final_20201012/to_test_cpu_important/test_bert.p"
    rl_all = "../testing_final_20201012/to_test_cpu_important/25_test_rl_all.p"
    bert = pickle.load(open(bert, "rb"))[0]
    rl_all = pickle.load(open(rl_all, "rb"))[0]
    for each in bert:
        if each[0] == "hmt_86d_19ae56f3ace2ddf09fd46bceb9d2e807e7677723":
            e = each
    #e = bert[10]

    id_, content, abs_, content_sent, abs_sent, _, _, probs = e
    for each in rl_all:
        if each[0] == id_:
            break
    id_rl, _, _ ,_ ,_, _, _, probs_rl = each
    print(id_, id_rl)
    print(len(probs_rl))

    probs = probs_rl

    total = {}
    for i, each in enumerate(content_sent):
        print(i, each)
        print(extractKeywords(each, 1))
        total[each] = i

    print("abstract: ")
    for i, each in enumerate(abs_sent):
        print(i, each)

    total["[]"] = "null"
    pairs, wordPairs = my_generate_mindmap_v(content_sent, probs)

    print()
    for each in pairs:
        start, end = each
        if start == []:
            start = "[]"
        print([total[start], total[end]])
