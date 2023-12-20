import sys
sys.path.append("..")
from evaluations import main
import numpy as np
from read_test_files import read_test_files
import argparse
import pickle


def get_myresults(data):
    data_dict = dict()
    for each in data:
        id_, sents = each
        prob_matrix = np.random.rand(len(sents), len(sents))
        data_dict[id_] = [sents, prob_matrix]
    return data_dict


parser = argparse.ArgumentParser(description="Process some hyper parameters")
parser.add_argument("--model_type", dest="model_type", default="gcn")
parser.add_argument("--seed", dest="seed", default=25, type=int)
parser.add_argument("--model_name", dest="model_name", default="20230618")
parser.add_argument("--longer", dest="longer", default=False, type=bool)
config = parser.parse_args()
print(config)
with open('log.txt', 'a+', encoding='utf-8') as f:
    f.write(f"\nseed = {config.seed}, model_type = {config.model_type}")

if __name__ == '__main__':
    benchmarks = "../testing_final_20201012/testing_data_hmt_20201012_final/a_labeling_test/"
    my_result_path = f"../testing_final_20201012/to_test/{config.seed}_test_{config.model_type}_{config.model_name}.p"

    my = pickle.load(open(my_result_path, "rb"))[0]

    my_results = dict()
    if config.longer:
        longer_results = dict()

    for e in my:
        id_, content, abs_, content_sent, abs_sent, _, _, probs = e
        # print(type(probs))
        # if config.model_type == "last_h" or config.model_type == "self_attention":
        probs = probs.cpu().detach().numpy()
        if not config.longer:
            my_results[id_] = [content_sent, probs]
        else:
            if len(content_sent) <= 25:
                my_results[id_] = [content_sent, probs]
            else:
                longer_results[id_] = [content_sent, probs]

    ### testing
    #sents = ['hello',
    #         'world',
    #         'flying']
    #prob_matrix = np.random.rand(4, 4)
    #my_results['test'] = [sents, prob_matrix]

    sim_threshold = 0.8

    main(benchmarks, my_results, sim_threshold)
    if config.longer:
        print("\nfor longer articles:")
        main(benchmarks, longer_results, sim_threshold)
