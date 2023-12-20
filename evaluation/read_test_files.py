import os
import pickle
import nltk
# We use these to separate the summary sentences in the datafiles
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", '"', dm_single_close_quote, dm_double_close_quote, ")"]

def read_test_files(path):
    all_info = []
    count_all = 0
    for file_name in os.listdir(path):
        file_ = path + file_name
        id_ = file_name[0: file_name.find(".story")]
        content, abs_ = get_art_abs(file_)

        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        for r in content:
            if r.rstrip():
                sentences += sent_detector.tokenize(r.rstrip())

        all_info.append([id_, sentences])
    print("all ", count_all)
    return all_info

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    if line==None: print("None")
    return line + "."

def get_art_abs(story_file):
    lines = read_text_file(story_file)

    lines = [line.lower() for line in lines]
    lines = [fix_missing_period(line) for line in lines]

    article_lines = []
    highlights = []
    next_is_highlight = False
    #print(lines)
    for idx, line in enumerate(lines):
        #print(line)
        if line == "":
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights








