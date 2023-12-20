import os
import pickle
import json
import numpy as np


def read_test_set():
    path = "testing_final_20201012/processed/test_full.p"
    full_test_data = pickle.load(open(path, "rb"))[0]
    # print(full_test_data)
    for article in full_test_data:
        title = article[0]
        with open('testset/' + title + '.txt', "w", encoding="utf-8") as file:
            for index, sentence in enumerate(article[3], start=0):
                file.write(f"{index}. {sentence}\n")


if __name__ == '__main__':
    file_list = os.listdir('/home/zhaoxiaoqun/zzw/mindmap_20230618/chatgpt_rg')
    rg_dict = {}
    for file in file_list:
        if file == 'chatgpt_rg.p':
            continue
        # 打开文件以读取内容
        with open('/home/zhaoxiaoqun/zzw/mindmap_20230618/chatgpt_rg/' + file, 'r') as f:
            lines = f.readlines()

        # 处理每一行，删除井号及其后面的内容
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.split('#')[0].strip()  # 使用split以及strip来删除井号及其后面的内容
            if cleaned_line in ['', '[', ']'] or cleaned_line.startswith('#'):
                continue
            if cleaned_line[-1] == ',':
                cleaned_line = cleaned_line[:-1]

            values = json.loads(cleaned_line)
            values[len(cleaned_lines)] = 1
            cleaned_lines.append(values)

        rg_dict[file] = np.array(cleaned_lines)


    with open('/home/zhaoxiaoqun/zzw/mindmap_20230618/chatgpt_rg/chatgpt_rg.p', 'wb') as f:
        pickle.dump(rg_dict, f)

