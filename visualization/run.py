import argparse
import pickle
from mind_map_generation import my_generate_mindmap_v, getkeyword

parser = argparse.ArgumentParser(description="Process some hyper parameters")
parser.add_argument("--model_type", dest="model_type", default="rl_all")
config = parser.parse_args()
print(config)

if __name__ == '__main__':
    # my_result_path = "../testing_sequential_20201016/to_test/sequential_result.p"
    # my_result_path = "../testing_final_20201012/to_test_cpu_important/25_test_rl_all.p"
    my_result_path = "../testing_final_20201012/to_test/2_test_gcn_20230809.p"
    # my_result_path = "../testing_final_20201012/to_test_cpu_important/test_bert.p"
    rg_path = "../evaluation/chatgpt_rg.p"
    my = pickle.load(open(my_result_path, "rb"))[0]
    rgs = pickle.load(open(rg_path, "rb"))
    e = my[66]
    rg = rgs['hmt_3_345e55780cfba82701239860d86d6135694fe178.txt']

    # wordDict = {}
    # keywords = []
    # original_sentences = e[3]
    #
    # for i in range(len(original_sentences)):
    #     kwd, keywords, wordDict = getkeyword(wordDict, i, original_sentences[i], keywords)
    #     print(kwd)


    id_, content, abs_, content_sent, abs_sent, _, _, probs = e
    # if config.model_type == "last_h" or config.model_type == "self_attention":
    # probs = probs.cpu().detach().numpy()
    probs = rg


    total = {}
    for i, each in enumerate(content_sent):
        print(i, each)
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

