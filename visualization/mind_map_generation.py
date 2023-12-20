import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')
import rouge
import re
import rake
import numpy as np
from collections import defaultdict
import copy
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans


def clean_sentences(doc_sents, summary_sents):
    stemmer = SnowballStemmer("english")
    stemmed_sents = []
    for s in doc_sents + summary_sents:
        temp_s = re.sub("[^a-zA-Z]", " ", s)
        temp_s = temp_s.lower()
        words = temp_s.split()
        words = [stemmer.stem(w) for w in words if not w in stopwords.words("english")]
        stemmed_sents.append(words)
    return stemmed_sents[:len(doc_sents)], stemmed_sents[len(doc_sents):]

def rouge_sim(hl, sent, a, b):
    '''
    a=b=0: r1
    b=0, r1+r2
    '''
    def contains(small, big):
        for i in xrange(len(big) - len(small) + 1):
            for j in xrange(len(small)):
                if big[i + j] != small[j]:
                    break
            else:
                return i, i + len(small)
        return False

    if hl == []:
        r1 = 1.0
    elif sent == []:
        r1 = 0.
    else:
        count_r1 = 0
        for c in hl:
            if c in sent:
                count_r1 += 1
        r1 = count_r1 * 1.0 / len(hl)
    if a <= 0 or len(hl) <= 1:
        return r1

    count_r2 = 0
    for i in range(len(hl) - 2 + 1):
        if contains(hl[i: i+2], sent):
            count_r2 += 1
    r2 = count_r2 * 1.0 / (len(hl) - 2 + 1)
    if b <= 0:
        return r1 * a + (1 - a) * r2

    if hl == sent:
        return r1 * a + (1 - a - b) * r2 + b
    else:
        return r1 * a + (1 - a - b) * r2

def rouge_sim2(summary, hypothesis):
    if summary == []:
        summary = ""
    if hypothesis == []:
        hypothesis = ""
    evaluator = rouge.Rouge(metrics = ['rouge-n', 'rouge-l'],
                            max_n = 2
                            )
    r_1 = evaluator.get_scores(hypothesis, summary)['rouge-1']['f']
    r_2 = evaluator.get_scores(hypothesis, summary)['rouge-2']['f']
    r_l = evaluator.get_scores(hypothesis, summary)['rouge-l']['f']
    score = r_1 / 3 + r_2 / 3 + r_l / 3
    return score

def extractKeywords(sentences, numSents):
    rake_object = rake.Rake("SmartStoplist.txt")
    if numSents > 1:
        text = ' '.join(sentences)
    else:
        text = sentences

    keywords = rake_object.run(text)
    counter = 0
    score = -1
    for word in keywords:
        if counter == 0:
            score = word[1]
        elif score != word[1]:
            break
        counter += 1
    if counter == len(keywords):
        kwds = [text.split()]
    else:
        kwds = [word[0].split() for word in keywords[:counter]]
    return kwds

def findKeywordsInSent(sent, candidates):
    return extractKeywords(sent, 1), False

def getkeyword(wordDict, key, sentence, keywords):
    if key in wordDict.keys():
        return wordDict[key], keywords, wordDict
    else:
        kwd, flag = findKeywordsInSent(sentence, keywords)
        if len(kwd) == 0:
            kwd = sentence.split()
        return kwd, keywords, wordDict

def parse_docs(filename):
    with open(filename, 'r') as f:
        content = f.read()
        finder = re.compile('<highlight>.*?</highlight>', re.DOTALL)
        contents = re.findall(finder, content, flags=0)
        contents = [cont.replace('<highlight>', '') for cont in contents]
        contents = [cont.replace('</hightlight>', '') for cont in contents]

        finder = re.compile('<T.*?</T', re.DOTALL)
        contents = re.findall(finder, contents[0], flags=0)
        contents = [re.split('>', c) for c in contents]
        tags = [c[0] for c in contents]
        tags = [tag.replace('<T.', '').replace('<T', '').replace('>', '') for tag in tags]
        contents = [c[1].replace('</T', '') for c in contents]

        keywords = []
        sentences = contents

        tag_cont = dict()
        tag_word = dict()
        for t, c, s in zip(tags, contents, sentences):
            tag_cont[t] = c
            tag_word[t] = s
        pairs = []
        kwdPairs = []
        wordDict = dict()
        content = ''
        for t in tags:
            if '.' not in t:
                pairs.append([[], tag_cont[t]])

                kwd, keywords, wordDict = getkeyword(wordDict, t, tag_word[t], keywords)
                if type(kwd[0]) == list:
                    kwd = kwd[0]
                kwdPairs.append([[], kwd])
                keywords.append(kwd)
                wordDict[t] = kwd
            else:
                father = t[:-2]
                while father not in tag_cont.keys():
                    father = father[:-2]
                pairs.append([tag_cont[father], tag_cont[t]])

                if t in wordDict.keys():
                    childword = wordDict[t]
                else:
                    childword, keywords, wordDict = getkeyword(wordDict, t, tag_word[t], keywords)
                    selectedChild = None
                    if type(childword[0]) != list:
                        if childword  not in keywords:
                            selectedChild = childword
                    else:
                        for child in childword:
                            if child not in keywords:
                                selectedChild = child
                                break
                    if selectedChild == None:
                        continue

                    childword = selectedChild

                if father in wordDict.keys():
                    fatherword = wordDict[father]
                else:
                    fatherword, keywords, wordDict = getkeyword(wordDict, father, tag_word[father], keywords)
                    selectedChild = None
                    if type(fatherword[0]) != list:
                        if fatherword not in keywords:
                            selectedChild = fatherword
                    else:
                        for child in fatherword:
                            if child not in keywords:
                                selectedChild = child
                                break
                    if selectedChild == None:
                        continue
                    fatherword = selectedChild

                if fatherword == childword:
                    continue

                keywords.append(childword)
                wordDict[t] = childword
                wordDict[father] = fatherword
                keywords.append(fatherword)
                kwdPairs.append([fatherword, childword])
            newline = " ".join(wordDict[t])
            newline = t + ' ' + newline + '\n'
            content += newline
        #newfile = filename.replace('story', 'word')
        #with open(newfile, 'w') as nf:
        #    nf.write(content)
        print(keywords)
    return pairs, kwdPairs, len(contents)

def maximum_ranking(prob_matrix):
    """Desceding order"""
    np.fill_diagonal(prob_matrix, 1)
    prob_matrix = prob_matrix.sum(axis=1)
    indexes = prob_matrix.argsort()
    indexes = list(reversed(indexes))
    return indexes


def generating_relation_tree_bottom_up(prob_matrix, generating_threshold):
    '''
    A node is regarded as the father of another node once its generating prob is greater than the generating threshold
    '''
    indexes = maximum_ranking(prob_matrix)
    indexes = list(reversed(indexes))
    #print(prob_matrix)
    #print(indexes)

    sentences_count = len(prob_matrix)
    index_of_sents = range(sentences_count)

    father_to_child = defaultdict(list)
    child_to_father = defaultdict(list)

    for location, selected_index in enumerate(indexes[: -1]):
        #print("--------", selected_index)
        prob_matrix[selected_index] = np.zeros(sentences_count)
        graph_matrix = prob_matrix.T
        #print(graph_matrix)

        if selected_index in father_to_child:
            idxs = copy.deepcopy(father_to_child[selected_index])
            idxs.append(selected_index)

            selected_list = graph_matrix[selected_index]
            if selected_list.sum() <= 0:
                child_to_father[selected_index] = []
                continue

            fathers_of_selected_index = np.where(selected_list > generating_threshold)[0]
            father_of_selected_index = -1
            for father in fathers_of_selected_index:
                if father_of_selected_index < 0 or (
                    father in indexes[location + 1:] and indexes.index(father) < indexes.index(
                    father_of_selected_index)):
                    father_of_selected_index = father
            if father_of_selected_index < 0:
                continue

            if father_of_selected_index in father_to_child:
                father_to_child[father_of_selected_index] += idxs
            else:
                father_to_child[father_of_selected_index] = idxs

            for node in idxs:
                child_to_father[node].append(father_of_selected_index)
        else:
            selected_list = graph_matrix[selected_index]
            #print(selected_list)
            fathers_of_selected_index = np.where(selected_list > generating_threshold)[0]
            father_of_selected_index = -1
            for father in fathers_of_selected_index:
                if father_of_selected_index < 0 or (
                    father in indexes[location + 1:] and indexes.index(father) < indexes.index(father_of_selected_index)):
                    father_of_selected_index = father
                    #print(father)
            if father_of_selected_index < 0:
                father_of_selected_index = np.where(selected_list == np.max(selected_list))[0][0]

            father_to_child[father_of_selected_index].append(selected_index)
            child_to_father[selected_index].append(father_of_selected_index)
    return father_to_child, child_to_father, indexes



def generating_relation_tree_through_clustering2(prob_matrix, dist_threshold):
    T_threshold = 0.5

    def distCal(sent, sentSet):
        minDist = 99999
        for i in sentSet:
            dist = abs(prob_matrix[sent] - prob_matrix[i]).sum(axis=0)
            if dist < minDist:
                minDist = dist
        return minDist / prob_matrix.shape[1]

    def maximum_ranking(prob_matrix):
        '''Desceding order'''
        np.fill_diagonal(prob_matrix, 1)
        prob_vector = prob_matrix.sum(axis=1)
        indexes = prob_vector.argsort()
        indexes = list(reversed(indexes))

        return indexes, prob_vector

    def mergeListDict(new, old):
        if new == None and old == None:
            return None
        elif old == None:
            return copy.deepcopy(new)
        elif new == None:
            return copy.deepcopy(old)
        final = copy.deepcopy(new)
        for k1 in old.keys():
            if k1 in new.keys():
                final[k1] = old[k1] + new[k1]
            else:
                final[k1] = old[k1]
        return final

    def clustering(feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents):

        if feature_matrix.shape[0] <= 1:
            return [], None, None

        ward = AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity)
        ward.fit(feature_matrix)
        label = ward.labels_

        sub_index_of_sents = defaultdict(list)
        for k, v in enumerate(label):
            sub_index_of_sents[v].append(index_of_sents[k])

        new_selected_sents = []
        new_father_to_child = defaultdict(list)
        new_child_to_father = defaultdict(list)
        for k in sub_index_of_sents.keys():
            feature_matrix = prob_matrix.T[np.ix_(sub_index_of_sents[k])]

            indexes, generating_probs = maximum_ranking(feature_matrix.T)

            for i in indexes:
                if (i in sub_index_of_sents[k]) and (distCal(i, selected_sents) > dist_threshold) and (
                    len(sub_index_of_sents[k]) > 1) and (generating_probs[i] / feature_matrix.shape[0] > T_threshold):
                    temp_sent = i

                    new_selected_sents.append(temp_sent)
                    sub_index_of_sents[k].remove(temp_sent)
                    new_father_to_child[temp_sent] = sub_index_of_sents[k]
                    for c in sub_index_of_sents[k]:
                        new_child_to_father[c].append(temp_sent)
                    break

        final_selected_sents = []
        final_father_to_child = None
        final_child_to_father = None
        for k in sub_index_of_sents.keys():
            feature_matrix = prob_matrix.T[np.ix_(sub_index_of_sents[k])]
            connectivity = None
            new_new_selected_sents, new_new_father_to_child, new_new_child_to_father = clustering(
                feature_matrix, connectivity, sub_index_of_sents[k],
                mergeListDict(new_father_to_child, father_to_child),
                mergeListDict(new_child_to_father, child_to_father),
                selected_sents + new_selected_sents)

            final_selected_sents = final_selected_sents + new_new_selected_sents
            final_father_to_child = mergeListDict(final_father_to_child, new_new_father_to_child)
            final_child_to_father = mergeListDict(final_child_to_father, new_new_child_to_father)

        final_selected_sents = new_selected_sents + final_selected_sents
        final_father_to_child = mergeListDict(final_father_to_child, new_father_to_child)
        final_child_to_father = mergeListDict(final_child_to_father, new_child_to_father)

        return final_selected_sents, final_father_to_child, final_child_to_father

    indexes, generating_probs = maximum_ranking(prob_matrix)
    root = indexes[0]
    if generating_probs[root] / prob_matrix.shape[0] < T_threshold:
        return None, None, None
    selected_sents = [root]
    father_to_child = defaultdict(list)
    child_to_father = defaultdict(list)
    father_to_child[root] = indexes[1:]
    for i in indexes[1:]:
        child_to_father[i].append(root)
    index_of_sents = list(range(prob_matrix.shape[0]))
    index_of_sents.remove(root)

    feature_matrix = prob_matrix.T[np.ix_(index_of_sents)]
    connectivity = None
    final_selected_sents, final_father_to_child, final_child_to_father =  clustering(
        feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents)

    final_father_to_child = mergeListDict(final_father_to_child, father_to_child)
    final_child_to_father = mergeListDict(final_child_to_father, child_to_father)

    indexes = indexes[::-1]
    for k in final_father_to_child.keys():
        final_father_to_child[k] = final_father_to_child[k][::-1]
    for k in final_child_to_father.keys():
        final_child_to_father[k] = final_child_to_father[k][::-1]

    return final_father_to_child, final_child_to_father, indexes

def generating_relation_tree_through_clustering(prob_matrix, dist_threshold):
    def distCal(sent, sentSet):
        minDist = 99999
        for i in sentSet:
            dist = abs(prob_matrix[sent] - prob_matrix[i]).sum(axis=0)
            if dist < minDist:
                minDist = dist
        return minDist / prob_matrix.shape[1]

    indexes = maximum_ranking(prob_matrix)

    def mergeListDict(new, old):
        if new == None and old == None:
            return None
        elif old == None:
            return copy.deepcopy(new)
        elif new == None:
            return copy.deepcopy(old)
        final = copy.deepcopy(new)
        for k1 in old.keys():
            if k1 in new.keys():
                final[k1] = old[k1] + new[k1]
            else:
                final[k1] = old[k1]
        return final

    def clustering(feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents):
        if feature_matrix.shape[0] <= 1:
            return [], None, None

        ward = AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity)
        ward.fit(feature_matrix)
        label = ward.labels_

        sub_index_of_sents = defaultdict(list)
        for k, v in enumerate(label):
            sub_index_of_sents[v].append(index_of_sents[k])

        new_selected_sents = []
        new_father_to_child = defaultdict(list)
        new_child_to_father = defaultdict(list)
        for k in sub_index_of_sents.keys():
            temp_sent = None
            for i in indexes:
                if i in sub_index_of_sents[k]:
                    temp_sent = i
                    break
            if distCal(temp_sent, selected_sents) > dist_threshold and len(sub_index_of_sents[k]) > 1:
                new_selected_sents.append(temp_sent)
                sub_index_of_sents[k].remove(temp_sent)
                new_father_to_child[temp_sent] = sub_index_of_sents[k]
                for c in sub_index_of_sents[k]:
                    new_child_to_father[c].append(temp_sent)

        final_selected_sents = []
        final_father_to_child = None
        final_child_to_father = None
        for k in sub_index_of_sents.keys():
            feature_matrix = prob_matrix[np.ix_(sub_index_of_sents[k])]
            connectivity = prob_matrix[np.ix_(sub_index_of_sents[k], sub_index_of_sents[k])]
            new_new_selected_sents, new_new_father_to_child, new_new_child_to_father = clustering(
                feature_matrix, connectivity, sub_index_of_sents[k],
                mergeListDict(new_father_to_child, father_to_child),
                mergeListDict(new_child_to_father, child_to_father),
                selected_sents + new_selected_sents
            )

            final_selected_sents = final_selected_sents + new_new_selected_sents
            final_father_to_child = mergeListDict(final_father_to_child, new_new_father_to_child)
            final_child_to_father = mergeListDict(final_child_to_father, new_new_child_to_father)

        final_selected_sents = new_selected_sents + final_selected_sents
        final_father_to_child = mergeListDict(final_father_to_child, new_father_to_child)
        final_child_to_father = mergeListDict(final_child_to_father, new_child_to_father)
        return final_selected_sents, final_father_to_child, final_child_to_father

    root = indexes[0]
    selected_sents = [root]
    father_to_child = defaultdict(list)
    child_to_father = defaultdict(list)
    father_to_child[root] = indexes[1:]
    for i in indexes[1:]:
        child_to_father[i].append(root)
    index_of_sents = list(range(prob_matrix.shape[0]))
    index_of_sents.remove(root)

    feature_matrix = prob_matrix[np.ix_(index_of_sents)]
    connectivity = prob_matrix[np.ix_(index_of_sents, index_of_sents)]
    final_selected_sents, final_father_to_child, final_child_to_father = clustering(
        feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents
    )
    final_father_to_child = mergeListDict(final_father_to_child, father_to_child)
    final_child_to_father = mergeListDict(final_child_to_father, child_to_father)

    indexes = indexes[::-1]
    for k in final_father_to_child.keys():
        final_father_to_child[k] = final_father_to_child[k][::-1]
    for k in final_child_to_father.keys():
        final_child_to_father[k] = final_child_to_father[k][::-1]

    return final_father_to_child, final_child_to_father, indexes

def truncating_tree_through_local_keywords(father_to_child, child_to_father, indexes, original_sentences, numPairs,
                                           sim_threshold, length_threshold):
    '''
    sim_threshold: threshold to filter out redundent nodes when tree generating
    '''
    maxNumberOfChildren = 1000

    def simMeasure(pair, pairs):
        if len(pairs) == 0:
            return 0.
        maxSim = 0.
        for p in pairs:
            sim = rouge_sim2(pair[0], p[0])
            sim += rouge_sim2(pair[1], p[1])
            sim = sim * 1.0 / 2
            if sim > maxSim:
                maxSim = sim
        return maxSim

    keywords = []
    #parsed_sentences, _ = clean_sentences(original_sentences, [])
    parsed_sentences = original_sentences
    # 现在indexes是从小到大排的
    root = indexes[-1]
    next = father_to_child[root]

    pairs = []
    wordPairs = []
    wordDict = dict()

    pairs.append([[], parsed_sentences[root]])

    kwd, keywords, wordDict = getkeyword(wordDict, root, original_sentences[root], keywords)
    if type(kwd[0]) == list:
        kwd = kwd[0]
    wordPairs.append([[], kwd])
    wordDict[root] = kwd
    keywords.append(kwd)
    content = '>[%d] %d %s\n' % (len(child_to_father[root]), root, original_sentences[root])
    wordMap = ">[%d] %d %s\n" % (len(child_to_father[root]), root, kwd)

    nodes = father_to_child[root]
    pathLength = [(next, len(child_to_father[next])) for next in nodes]
    sortList = sorted(pathLength, key=lambda nodes: nodes[1])
    sortList = [s[0] for s in sortList][::-1]

    deepest = 0
    childCounter = defaultdict(list)
    # for s in sortList: # deep first
    for s in indexes[:-1][::-1]: # width first
        #flag = False

        fathers = list(reversed(child_to_father[s]))
        # fathers = child_to_father[s]
        if len(fathers) <= 0:
            continue
        father = fathers[0]
        indent = "-%d" % father
        counter = 1
        for i in fathers[1:]:
            if len(childCounter[father]) > maxNumberOfChildren:
                continue

            if len(pairs) == numPairs:
                break

            if i in wordDict.keys():
                indent += "-%d" % i
                father = i
                continue

            pair = [parsed_sentences[father], parsed_sentences[i]]
            if len(original_sentences[i]) < length_threshold:
                continue

            childword, keywords, wordDict = getkeyword(wordDict, i, original_sentences[i], keywords)
            fatherword, keywords, wordDict = getkeyword(wordDict, father, original_sentences[father], keywords)

            selectedChild = None
            if type(childword[0]) != list:
                if childword not in keywords:
                    selectedChild = childword
            else:
                for child in childword:
                    if child not in keywords:
                        selectedChild = child
                        break
            if selectedChild == None:
                continue

            if len(child_to_father[s]) >= deepest:
                deepest = len(child_to_father[s])

            content += "|-%s-->[%d] %d %s \n" % (indent, len(child_to_father[i]), i, original_sentences[i])
            wordMap += "|-%s-->[%d] %d %s \n" % (indent, len(child_to_father[i]), i, selectedChild)
            keywords.append(selectedChild)
            indent += "-%d" % i
            pairs.append(pair)
            wordDict[i] = selectedChild
            wordPairs.append([fatherword, selectedChild])
            childCounter[father].append(i)
            father = i
            counter += 1
        if len(pairs) == numPairs:
            break

        if len(childCounter[father]) >= maxNumberOfChildren:
            continue

        pair = [parsed_sentences[father], parsed_sentences[s]]
        if len(original_sentences[s]) < length_threshold:
            continue

        fatherword, keywords, wordDict = getkeyword(wordDict, father, original_sentences[father], keywords)
        childword, keywords, wordDict = getkeyword(wordDict, s, original_sentences[s], keywords)

        selectedChild = None
        if type(childword[0]) != list:
            if childword not in keywords:
                selectedChild = childword
        else:
            for child in childword:
                if child not in keywords:
                    selectedChild = child
                    break
        if selectedChild == None:
            continue

        keywords.append(selectedChild)
        wordDict[s] = selectedChild
        wordPairs.append([fatherword, selectedChild])
        content += "|-%s-->[%d] %d %s\n" % (indent, len(child_to_father[s]), s, original_sentences[s])
        wordMap += "|-%s-->[%d] %d %s\n" % (indent, len(child_to_father[s]), s, selectedChild)
        pairs.append(pair)

        childCounter[father].append(s)

        if len(child_to_father[s]) > deepest:
            deepest =  len(child_to_father[s])
    #print(content)
    return pairs, wordPairs

def truncating_tree_through_local_keywords_wy_old(father_to_child, child_to_father, indexes, original_sentences, numPairs,
                                           sim_threshold, length_threshold):
    '''
    sim_threshold: threshold to filter out redundent nodes when tree generating
    length_threshold: threshold to determin the minimum length a sentence selected into a mindmap
    '''
    maxNumberOfChildren = 1000

    def simMeasure(pair, pairs):
        if len(pairs) == 0:
            return 0.
        maxSim = 0.
        for p in pairs:
            sim = rouge_sim2(pair[0], p[0])
            sim += rouge_sim2(pair[1], p[1])
            sim = sim * 1.0 / 2
            if sim > maxSim:
                maxSim = sim
        return maxSim

    keywords = []
    #parsed_sentences, _ = clean_sentences(original_sentences, [])
    parsed_sentences = original_sentences
    # 现在indexes是从小到大排的
    root = indexes[-1]
    next = father_to_child[root]

    pairs = []
    wordPairs = []
    wordDict = dict()

    pairs.append([[], parsed_sentences[root]])

    kwd, keywords, wordDict = getkeyword(wordDict, root, original_sentences[root], keywords)
    if type(kwd[0]) == list:
        kwd = kwd[0]
    wordPairs.append([[], kwd])
    wordDict[root] = kwd
    keywords.append(kwd)
    content = '>[%d] %d %s\n' % (len(child_to_father[root]), root, original_sentences[root])
    wordMap = ">[%d] %d %s\n" % (len(child_to_father[root]), root, kwd)

    nodes = father_to_child[root]
    pathLength = [(next, len(child_to_father[next])) for next in nodes]
    sortList = sorted(pathLength, key=lambda nodes: nodes[1])
    sortList = [s[0] for s in sortList][::-1]

    deepest = 0
    childCounter = defaultdict(list)
    # for s in sortList: # deep first
    for s in indexes[:-1][::-1]: # width first
        #flag = False

        fathers = list(reversed(child_to_father[s]))
        # fathers = child_to_father[s]
        if len(fathers) <= 0:
            continue
        father = fathers[0]
        indent = "-%d" % father
        counter = 1
        for i in fathers[1:]:
            # when the children of a node is more than 5, continue
            if len(childCounter[father]) > maxNumberOfChildren:
                continue

            if len(pairs) == numPairs:
                break

            if i in wordDict.keys():
                indent += "-%d" % i
                father = i
                continue

            pair = [parsed_sentences[father], parsed_sentences[i]]
            if len(original_sentences[i]) < length_threshold or simMeasure(pair, pairs) >  sim_threshold:
                continue

            childword, keywords, wordDict = getkeyword(wordDict, i, original_sentences[i], keywords)
            fatherword, keywords, wordDict = getkeyword(wordDict, father, original_sentences[father], keywords)

            selectedChild = None
            if type(childword[0]) != list:
                if childword not in keywords:
                    selectedChild = childword
            else:
                for child in childword:
                    if child not in keywords:
                        selectedChild = child
                        break
            if selectedChild == None:
                continue

            if len(child_to_father[s]) >= deepest:
                deepest = len(child_to_father[s])

            content += "|-%s-->[%d] %d %s \n" % (indent, len(child_to_father[i]), i, original_sentences[i])
            wordMap += "|-%s-->[%d] %d %s \n" % (indent, len(child_to_father[i]), i, selectedChild)
            keywords.append(selectedChild)
            indent += "-%d" % i
            pairs.append(pair)
            wordDict[i] = selectedChild
            wordPairs.append([fatherword, selectedChild])
            childCounter[father].append(i)
            father = i
            counter += 1
        if len(pairs) == numPairs:
            break

        if len(childCounter[father]) >= maxNumberOfChildren:
            continue

        pair = [parsed_sentences[father], parsed_sentences[s]]
        if len(original_sentences[s]) < length_threshold or simMeasure(pair, pairs) > sim_threshold:
            continue

        fatherword, keywords, wordDict = getkeyword(wordDict, father, original_sentences[father], keywords)
        childword, keywords, wordDict = getkeyword(wordDict, s, original_sentences[s], keywords)

        selectedChild = None
        if type(childword[0]) != list:
            if childword not in keywords:
                selectedChild = childword
        else:
            for child in childword:
                if child not in keywords:
                    selectedChild = child
                    break
        if selectedChild == None:
            continue

        keywords.append(selectedChild)
        wordDict[s] = selectedChild
        wordPairs.append([fatherword, selectedChild])
        content += "|-%s-->[%d] %d %s\n" % (indent, len(child_to_father[s]), s, original_sentences[s])
        wordMap += "|-%s-->[%d] %d %s\n" % (indent, len(child_to_father[s]), s, selectedChild)
        pairs.append(pair)

        childCounter[father].append(s)

        if len(child_to_father[s]) > deepest:
            deepest =  len(child_to_father[s])
    print(wordMap)
    return pairs, wordPairs


def generating_relation_tree_through_clustering_hmt(prob_matrix):
    T_threshold = 0.5

    def maximum_ranking(prob_matrix):
        '''Desceding order'''
        np.fill_diagonal(prob_matrix, 1)
        prob_vector = prob_matrix.sum(axis=1)
        indexes = prob_vector.argsort()
        indexes = list(reversed(indexes))

        return indexes, prob_vector

    def mergeListDict(new, old):
        if new == None and old == None:
            return None
        elif old == None:
            return copy.deepcopy(new)
        elif new == None:
            return copy.deepcopy(old)
        final = copy.deepcopy(new)
        for k1 in old.keys():
            if k1 in new.keys():
                final[k1] = old[k1] + new[k1]
            else:
                final[k1] = old[k1]
        return final

    def clustering(feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents):

        if feature_matrix.shape[0] <= 1:
            return [], None, None

        ward = KMeans(n_clusters=2)
        # Kmeans
        ward.fit(feature_matrix)
        label = ward.labels_

        sub_index_of_sents = defaultdict(list)
        for k, v in enumerate(label):
            sub_index_of_sents[v].append(index_of_sents[k])

        #print(sub_index_of_sents)
        new_selected_sents = []
        new_father_to_child = defaultdict(list)
        new_child_to_father = defaultdict(list)
        for k in sub_index_of_sents.keys():
            feature_matrix = prob_matrix.T[np.ix_(sub_index_of_sents[k])]

            indexes, generating_probs = maximum_ranking(feature_matrix.T)

            for i in indexes:
                if (i in sub_index_of_sents[k]) and (len(sub_index_of_sents[k]) > 0) and (
                    len(sub_index_of_sents[k]) == 1 or generating_probs[i] / feature_matrix.shape[0] > T_threshold):
                    temp_sent = i
                    #print("clustering ", i)

                    new_selected_sents.append(temp_sent)
                    sub_index_of_sents[k].remove(temp_sent)
                    new_father_to_child[temp_sent] = sub_index_of_sents[k]
                    for c in sub_index_of_sents[k]:
                        new_child_to_father[c].append(temp_sent)
                    break

        final_selected_sents = []
        final_father_to_child = None
        final_child_to_father = None
        for k in sub_index_of_sents.keys():
            feature_matrix = prob_matrix.T[np.ix_(sub_index_of_sents[k])]
            connectivity = None
            new_new_selected_sents, new_new_father_to_child, new_new_child_to_father = clustering(
                feature_matrix, connectivity, sub_index_of_sents[k],
                mergeListDict(new_father_to_child, father_to_child),
                mergeListDict(new_child_to_father, child_to_father),
                selected_sents + new_selected_sents)

            final_selected_sents = final_selected_sents + new_new_selected_sents
            final_father_to_child = mergeListDict(final_father_to_child, new_new_father_to_child)
            final_child_to_father = mergeListDict(final_child_to_father, new_new_child_to_father)

        final_selected_sents = new_selected_sents + final_selected_sents
        final_father_to_child = mergeListDict(final_father_to_child, new_father_to_child)
        final_child_to_father = mergeListDict(final_child_to_father, new_child_to_father)

        return final_selected_sents, final_father_to_child, final_child_to_father

    indexes, generating_probs = maximum_ranking(prob_matrix)
    root = indexes[0]
    #print('root ', root)
    #if generating_probs[root] / prob_matrix.shape[0] < T_threshold:
    #    return None, None, None
    selected_sents = [root]
    father_to_child = defaultdict(list)
    child_to_father = defaultdict(list)
    father_to_child[root] = indexes[1:]
    for i in indexes[1:]:
        child_to_father[i].append(root)
    index_of_sents = list(range(prob_matrix.shape[0]))
    index_of_sents.remove(root)

    feature_matrix = prob_matrix.T[np.ix_(index_of_sents)]
    connectivity = None
    final_selected_sents, final_father_to_child, final_child_to_father = clustering(
        feature_matrix, connectivity, index_of_sents, father_to_child, child_to_father, selected_sents)

    final_father_to_child = mergeListDict(final_father_to_child, father_to_child)
    final_child_to_father = mergeListDict(final_child_to_father, child_to_father)

    indexes = indexes[::-1]
    for k in final_father_to_child.keys():
        final_father_to_child[k] = final_father_to_child[k][::-1]
    for k in final_child_to_father.keys():
        final_child_to_father[k] = final_child_to_father[k][::-1]

    return final_father_to_child, final_child_to_father, indexes


def my_generate_mindmap(o, prob_matrix, numPaires, sim_threshold, length_threshold):
    father_to_child, child_to_father, indexes = generating_relation_tree_through_clustering_hmt(prob_matrix)
    pairs, wordPairs = truncating_tree_through_local_keywords(father_to_child, child_to_father, indexes, o, numPaires, sim_threshold, length_threshold)

    return pairs, wordPairs

def my_generate_mindmap_v(o, prob_matrix):
    father_to_child, child_to_father, indexes = generating_relation_tree_through_clustering_hmt(prob_matrix)
    print(father_to_child)
    print()
    print(child_to_father)
    pairs, wordPairs = truncating_tree_through_local_keywords_wy_old(father_to_child, child_to_father, indexes, o, 10, 0, 10)

    return pairs, wordPairs
