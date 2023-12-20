from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from os import listdir
from os.path import isfile, join
import re
import rake
import pickle
from mind_map_generation import *
import rouge

def process_wordPairs(data):
    new_data = []
    for each in data:
        new_each = []
        one, two = each
        if len(one) == 0:
            new_each.append("")
        else:
            new_each.append(" ".join(one))

        if len(two) == 0:
            new_each.append("")
        else:
            new_each.append(" ".join(two))

        new_data.append(new_each)
    return new_data


def compare_method(pairs, pairs2):
    sim = 0.0
    for i in range(len(pairs2)):
        # msp = ['', '']
        # old code
        msp = pairs[1]
        if i == 0:
            continue
        found = False
        for j in range(len(pairs)):
            # modified
            if j == 0:
                continue
            first = rouge_sim2(pairs2[i][0], msp[0]) + rouge_sim2(pairs2[i][1], msp[1])
            second = rouge_sim2(pairs2[i][0], pairs[j][0]) + rouge_sim2(pairs2[i][1], pairs[j][1])
            if first < second:
                msp = pairs[j]
                max_index = j
                found = True
        cur_sim = rouge_sim2(pairs2[i][0], msp[0])
        cur_sim += rouge_sim2(pairs2[i][1], msp[1])

        sim += cur_sim / 2

        if found:
            del pairs[max_index]
    return sim

def main(benchmarks, my_results, sim_threshold):
    cc = listdir(benchmarks)
    totalSim = 0
    totalSim_word = 00
    evaluator_number = 0
    cc.sort()
    for idx, target in enumerate(cc):
        if target.find('.story') >= 0:
            print(target)
            # if target == '12.story':
            #     continue
            pairs2, wordPairs2, length_threshold = parse_docs(join(benchmarks, target))
            # print(pairs2)
            # print(wordPairs2)

            target_id = target[0: target.find('.story')]
            sents, prob_matrix = my_results[target_id]

            pairs, wordPairs = my_generate_mindmap(sents, prob_matrix, len(pairs2), sim_threshold, length_threshold)
            # print(f'pairs = {pairs}')
            tmp_pairs = pairs[:]
            # print(f'tmp_pairs = {tmp_pairs}')
            sim = compare_method(tmp_pairs, pairs2)


            wordPairs = process_wordPairs(wordPairs)
            wordPairs2 = process_wordPairs(wordPairs2)
            sim_word = compare_method(wordPairs, wordPairs2)
            print(sim/len(pairs2), sim_word/len(wordPairs2))

            totalSim += sim / len(pairs2)
            totalSim_word += sim_word / len(wordPairs2)
            evaluator_number += 1

    print("final result for", evaluator_number, " files")
    print(str(totalSim / evaluator_number))
    print("key word: ", str(totalSim_word / evaluator_number))

    # modified
    return totalSim / evaluator_number, totalSim_word / evaluator_number









