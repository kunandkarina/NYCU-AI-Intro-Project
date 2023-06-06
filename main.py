from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import re
import os
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
from glob import glob
from tensorflow.keras.models import load_model
from tqdm import tqdm
import bert
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from train import convert_sentences_to_features

# num = 0
# directory = 'aclImdb/test/neg'
# pred_sentences1 = []
# files = os.listdir(directory)
# for file in files:
#     num += 1
#     if(num > 200):
#         break
#     with open(directory + "/" + file, 'r', encoding = 'utf-8') as f:
#         data = f.read()
#         pred_sentences1.append(data)
#         f.close()

# directory = 'aclImdb/test/pos'
# pred_sentences2 = []
# num = 0
# files = os.listdir(directory)
# for file in files:
#     num += 1
#     if(num > 200):
#         break
#     with open(directory + "/" + file, 'r', encoding = 'utf-8') as f:
#         data = f.read()
#         pred_sentences2.append(data)
#         f.close()


# model = TFBertForSequenceClassification.from_pretrained('bert_fine_tuning') #bert model after fine tuning
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# tf_batch1 = tokenizer(pred_sentences1, max_length=128, padding=True, truncation=True, return_tensors='tf')
# tf_outputs1 = model(tf_batch1)
# tf_predictions1 = tf.nn.softmax(tf_outputs1[0], axis=-1) #linear classifier
# labels = ['Negative','Positive']
# label1 = tf.argmax(tf_predictions1, axis=1)
# label1 = label1.numpy()
# tf_batch2 = tokenizer(pred_sentences2, max_length=128, padding=True, truncation=True, return_tensors='tf')
# tf_outputs2 = model(tf_batch2)
# tf_predictions2 = tf.nn.softmax(tf_outputs2[0], axis=-1)
# labels = ['Negative','Positive']
# label2 = tf.argmax(tf_predictions2, axis=1)
# label2 = label2.numpy()

# total = 0
# correct = 0
# for i in range(len(pred_sentences1)):
#     total += 1
#     if(0 == label1[i]):
#         correct += 1
# for i in range(len(pred_sentences2)):
#     total += 1
#     if(1 == label2[i]):
#         correct += 1
# print(f'total {total} comments, correctly identify {correct} comments, accuracy is {correct / total}')

# def create_tonkenizer(bert_layer):
#     """Instantiate Tokenizer with vocab"""
#     vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
#     do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() 
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     print("Vocab size:", len(tokenizer.vocab))
#     return tokenizer

# def get_ids(tokens, tokenizer, MAX_SEQ_LEN):
#     """Token ids from Tokenizer vocab"""
#     token_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
#     return input_ids

# def get_masks(tokens, MAX_SEQ_LEN):
#     """Masks: 1 for real tokens and 0 for paddings"""
#     return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))

# def get_segments(tokens, MAX_SEQ_LEN):
#     """Segments: 0 for the first sequence, 1 for the second"""  
#     segments = []
#     current_segment_id = 0
#     for token in tokens:
#         segments.append(current_segment_id)
#         if token == "[SEP]":
#             current_segment_id = 1
#     return segments + [0] * (MAX_SEQ_LEN - len(tokens))

# def create_single_input(sentence, tokenizer, max_len):
#     """Create an input from a sentence"""
#     stokens = tokenizer.tokenize(sentence)
#     stokens = stokens[:max_len] # max_len = MAX_SEQ_LEN - 2, why -2 ? ans: reserved for [CLS] & [SEP]
#     stokens = ["[CLS]"] + stokens + ["[SEP]"]
#     return get_ids(stokens, tokenizer, max_len+2), get_masks(stokens, max_len+2), get_segments(stokens, max_len+2)

# def convert_sentences_to_features(sentences, tokenizer, MAX_SEQ_LEN):
#     """Convert sentences to features: input_ids, input_masks and input_segments"""
#     input_ids, input_masks, input_segments = [], [], []
#     for sentence in tqdm(sentences, position=0, leave=True):
#       ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN-2) # why -2 ? ans: reserved for [CLS] & [SEP]
#       # assert len(ids) == MAX_SEQ_LEN
#       # assert len(masks) == MAX_SEQ_LEN
#       # assert len(segments) == MAX_SEQ_LEN
#       input_ids.append(ids)
#       input_masks.append(masks)
#       input_segments.append(segments)
#     return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_experimental_option("prefs", {"profile.password_manager_enabled": False, "credentials_enable_service": False})
options.chrome_executable_path = 'chromedriver.exe'

url = input("please the link of a movie review: ")
driver=webdriver.Chrome(options=options)
driver.get(url)
page = 1
# IMDB中每個頁面只有25則評論，因此我們必須翻10頁來取得200筆以上的資訊
while page < 10:
    try:
        # 用css_selector找尋'load-more-trigger'的位置
        css_selector = 'load-more-trigger'
        driver.find_element(By.ID, css_selector).click()
        time.sleep(3)
        page += 1
    except:
        break
# 尋找class = review-container的標籤
review = driver.find_elements(By.CLASS_NAME, 'review-container')
# 儲存星星數與評價的list
rating = []
lis = []
cnt = 0
# 設定最多找200筆資訊
for n in range(0,250):
    try:
        if cnt >=200:
            break
        # 用戶評論必須同時具備rating和title的資料，否則略過並尋找下一筆
        frating = review[n].find_element(By.CLASS_NAME, 'rating-other-user-rating').text
        flist = review[n].find_element(By.CLASS_NAME, 'title').text

        rating.append(frating)
        lis.append(flist)
        cnt += 1
    except:
        continue
# 將rating的資料從string轉成int
for j in range(len(rating)):
    rating[j] = rating[j].replace('/10', "")
    rating[j] = int(rating[j])

model = load_model('bert_fine_tuning') #bert model after fine tuning
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tf_batch = convert_sentences_to_features(lis, tokenizer, 128)
tf_outputs = model(tf_batch)
tf_predictions = model.predict(tf_batch) #linear classifier
labels = ['Negative','Positive']
label1 = tf.argmax(tf_predictions, axis=1)
label1 = label1.numpy()
total = 0
positive = 0
for i in range(len(lis)):
    total += 1
    if(label1[i] == 1):
        positive += 1

print(f'In total {total} comments, {positive} comments are positive, {100 * positive / total}% people think it is a good movie')
print(f'averge rate of 200 comments is {sum(rating) / len(rating)}')


            
    
