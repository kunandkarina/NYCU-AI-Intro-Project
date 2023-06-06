path_pos = "aclImdb/train/pos"
path_neg = "aclImdb/train/neg"
path_pos_test = "aclImdb/test/pos"
path_neg_test = "aclImdb/test/neg"
import os
def read_files_to_list(path):
    docs = []
    # 借助os.listdir找出特定folder下所有的files
    files = os.listdir(path)
    # print(files)
    k = 0
    for file in files:  
        k += 1
    # 再把path 和 file names join起來，就可以得到我們要的檔案位置
        with open(os.path.join(path, file),encoding="utf-8") as f:
            docs.append(f.read())
        # if (k == 2000) : break
    return docs
pos_file_list = read_files_to_list(path_pos)
neg_file_list = read_files_to_list(path_neg)
print(len(pos_file_list) + len(neg_file_list))
pos_test_list = read_files_to_list(path_pos_test)
neg_test_list = read_files_to_list(path_neg_test)
print(len(pos_test_list) + len(neg_test_list))

with open('train_ds.txt', 'w', encoding="utf-8") as file:
    for data in pos_file_list:
        file.write("1\t" + data + "\n")
    for data in neg_file_list:
        file.write("0\t" + data + "\n")
print("train finished")
with open("test_ds.txt", "w", encoding="utf-8") as file:
    for data in pos_test_list:
        file.write("1\t" + data + "\n")
    for data in neg_test_list:
        file.write("0\t" + data + "\n")
print("test finished")