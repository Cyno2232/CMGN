import os
import pickle
import nltk
import re

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

dm_single_close_quote = u'\u2019' #unicode
dm_double_close_quote = u'\u201d'

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


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

def nltk_tokenize(contents):
    tokens = []
    for r in contents:
        tokens += [nltk.word_tokenize(clean_str(r))]
    return tokens

def sentence_separate_and_tokenize(contents):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    for r in contents:
        if r.rstrip():
            sentences += sent_detector.tokenize(r.rstrip())
    return sentences

def process(data):
    all_token_data = []
    for each in data:
        id_, content, abstract = each
        content_token = sentence_separate_and_tokenize(content)
        abstract_token = sentence_separate_and_tokenize(abstract)
        cur = [id_, content, abstract, content_token, abstract_token]
        all_token_data.append(cur)
    return all_token_data

def split_total(path):
    all_info = []
    count_all = 0
    for file_name in os.listdir(path):
        file_ = path + file_name
        id_ = file_name[0: file_name.find(".story")]
        content, abs_ = get_art_abs(file_)
        count_all += 1
        if len(content) == 0:
            print("empty content")
        if len(content) > 0:
            content_sentences = sentence_separate_and_tokenize(content)
            abstract_sentences = sentence_separate_and_tokenize(abs_)
            content_tokens = nltk_tokenize(content_sentences)
            abstract_tokens = nltk_tokenize(abstract_sentences)
            all_info.append([id_, content, abs_, content_sentences, abstract_sentences, content_tokens, abstract_tokens])
    print("all ", count_all)
    print("no empty", len(all_info))
    #pickle.dump([all_info], open("processed/dev_full.p", "wb"))

def read_text_file(text_files):
    lines = []
    with open(text_files, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."

def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends  lines that are missing them (this is a problem in the dataset because many image captions don't end in periods;
    lines = [fix_missing_period(line) for line in lines]

    # separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            article_lines.append(line)
    return article_lines, highlights

if __name__ == '__main__':
    path = "testing_data_hmt_20201012_final/dev_full_original/"
    split_total(path)
