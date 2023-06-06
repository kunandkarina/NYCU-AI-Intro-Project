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
        # if (k == 10) : break
    return docs

pos_file_list = read_files_to_list(path_pos)
neg_file_list = read_files_to_list(path_neg)
pos_test_list = read_files_to_list(path_pos_test)
neg_test_list = read_files_to_list(path_neg_test)
import pandas as pd
# 把 positive的文章list 和 negative的文章list串接再一起，在和他們對應的label zip再一起，變成 
# ((positive article, 1)
#  (positive article, 1)
#  ...
#  (negative article, 0))
imdb_df = pd.DataFrame(data = zip(pos_file_list +neg_file_list, ['pos'] * len(pos_file_list) + ['neg'] * len(neg_file_list)))
imdb_df.columns = ['text', 'label']

imdb_test = pd.DataFrame(data = zip(pos_test_list + neg_test_list, ['pos']*len(pos_test_list) + ['neg']*len(neg_test_list)))
imdb_test.columns = ['text', 'label']
print("data load successfully.\n")
# print(imdb_df.head())
# print(imdb_df.tail())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 初始化vectorizer, 使用的是bag-of-word 最基礎的 CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
# stop_words="english"

# 將 text 轉換成 bow 格式
text = vectorizer.fit_transform(imdb_df['text'])
X_train, X_test, y_train, y_test = train_test_split(text, imdb_df['label'], test_size=0.2, random_state=0)
# 實例化(Instantiate) 這個 Naive Bayes Classifier
import time
start_time = time.time()
string = ("1 Naive Bayes Classifier\n" + 
"2 Decision Tree Classifier\n" +
"3 Random Forest Classifier")
print(string)
train_model = input("please enter the number to select the training model: ")
if(train_model == "1"):
    print("your training model is Naive Bayes Classifier\n")
    model = MultinomialNB()
elif(train_model == "2"):
    print("your training model is Decision Tree Classifier\n")
    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
else:
    print("your training model is Random Forest Classifier\n")
    model = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

model.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
print("predict: ", model.predict(X_test))
print("accuracy: ", model.score(X_test, y_test))

# test_text_pos = vectorizer.transform(['This movie is not interesting as what I had thought.'])
# print(MNB_model.predict(test_text_pos))
from sklearn.metrics import accuracy_score
answers = []
predicts = []
for i in range(len(pos_test_list+neg_test_list)) :
    if(i < len(pos_test_list)):
        answers.append('pos')
        test_text = vectorizer.transform([pos_test_list[i]])
        predicts.append(model.predict(test_text))
    else:
        answers.append('neg')
        test_text = vectorizer.transform([neg_test_list[i-len(pos_test_list)]])
        predicts.append(model.predict(test_text))
# print(accuracy_score(answers, predicts))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(answers, predicts)
TP_NB = cm[1][1]
TN_NB = cm[0][0]
FP_NB = cm[1][0]
FN_NB = cm[0][1]
Accuracy_NB = (TP_NB + TN_NB) / (TP_NB + TN_NB + FP_NB + FN_NB) 
Precision_NB = TP_NB / (TP_NB + FP_NB)
Recall_NB = TP_NB / (TP_NB + FN_NB)

print("accuracy is ", Accuracy_NB)
print("precision is ", Precision_NB)
print("recall is ", Recall_NB)
# print("confusion matrix is ", cm)
