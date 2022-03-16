# -*- coding: utf-8 -*-
"""3.1 국립국어원(NIKL) 개체명 인식.ipynb
"""

import json
import pickle

with open('C:/Users/woori/Downloads/국립국어원 개체명 분석 말뭉치(버전 1.0)/NXNE1902008030.json', 'r', encoding='UTF-8') as f:
    NXNE1902008030 = json.load(f) # dict_keys(['id', 'metadata', 'document'])
    
nilk = []
for dct in NXNE1902008030['document']:
    nilk.append(dct['sentence'])
    
tagged_nilk = []
for lst in nilk:
    for dct in lst:
        tmp_lst     = []
        tagged_word = []        
        if len(dct['NE']) > 0:
            for tmp in dct['NE']:
                tmp_lst.append((tmp['form'],tmp['label']))
        for word in dct['word']:
            tagged = word['form']
            i      = 0
            for ner in tmp_lst:
                i += 1
                if len(ner[0]) == len(tagged):
                    if ner[0] == tagged:
                        tagged_word.append([tagged, ner[1]])
                        break
                elif len(ner[0]) < len(tagged):
                    if ner[0] == tagged[0:len(ner[0])]:
                        tagged_word.append([tagged[0:len(ner[0])], ner[1]])
                        tagged_word.append([tagged[len(ner[0]):len(tagged)], 'O'])
                        break
                if i == len(tmp_lst):
                    tagged_word.append([tagged, 'O'])
        tagged_nilk.append(tagged_word) 

with open('Y:/SeeValue/ipynb/Python/Project/BigKinds/data/tagged_nilk.pickle', 'wb') as f:
    pickle.dump(tagged_nilk, f)     
    
type(tagged_nilk), len(tagged_nilk)

"""![image.png](attachment:image.png)
### 국립국어원 개체명 말뭉치분류 (2019)
PS(Person) / LC(Location) / OG(Organization) / AF(Artifacts) / DT(Date) / TI(Time) / CV(Civilization) / AM(Animal) / PT(Plant) / QT(Quantity) / FD(Study_Field) / TR(Theory) / EV(Event) / MT(Material) / TM(Term)

![image.png](attachment:image.png)
"""

import numpy             as np
import matplotlib.pyplot as plt
import nltk

from tqdm                                    import tqdm
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils                  import to_categorical
from sklearn.model_selection                 import train_test_split

# 토큰화에 품사 태깅이 된 데이터 받아오기 ( 국립국어원(2019))
with open('/SeeValue/ipynb/Python/Project/BigKinds/data/tagged_nilk.pickle', 'rb') as f:
    tagged_sentences = pickle.load(f) # 단 한줄씩 읽어옴
    
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences), '\n')
print(tagged_sentences[0:2], '\n')
print(tagged_sentences[45])

# 훈련을 시키려면 훈련 데이터에서 단어에 해당되는 부분과 품사 태깅 정보에 해당되는 부분을 분리
# [('Pierre', 'NNP'), ('Vinken', 'NNP')] -> Pierre과 Vinken을 같이 저장하고, NNP와 NNP를 같이 저장
sentences = [] # 단어정보 저장
pos_tags  = [] # 태깅정보 저장

for tagged_sentence in tqdm(tagged_sentences): # 150,082 개의 문장 샘플을 1개씩 불러온다.
    if len(tagged_sentence) > 0:
        sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 품사 태깅 정보들은 tag_info에 저장한다.
        sentences.append(list(sentence))           # 각 샘플에서 단어 정보만 저장한다.
        pos_tags.append(list(tag_info))            # 각 샘플에서 품사 태깅 정보만 저장한다.
    
print(len(sentences), sentences[0], '\n')
print(len(pos_tags), pos_tags[0])

print('샘플의 최대 길이 : %d' % max(len(l) for l in sentences))
print('샘플의 최소 길이 : %d' % min(len(l) for l in sentences))
print('샘플의 평균 길이 : %f' % (sum(map(len, sentences))/len(sentences)))
plt.figure(figsize=(8, 6))
plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def tokenize(samples): 
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(samples) # samples에 있는 모든 단어들을 이용하여 단어 사전를 만듦. 인덱스, 문장 내외 등장 횟수 등
  return tokenizer

# .word_counts, .word_docs, .word_index 등 속성 사용 가능
src_tokenizer = tokenize(sentences) # 
tag_tokenizer = tokenize(pos_tags)  #  

vocab_size = len(src_tokenizer.word_index) + 1 # + 1
tag_size   = len(tag_tokenizer.word_index) + 1 # + 1 
print('Type                  :', type(src_tokenizer))
print('단어 집합의 크기      : {}'.format(vocab_size))
print('태깅 정보 집합의 크기 : {}'.format(tag_size))

print(dir(src_tokenizer))

X_train = src_tokenizer.texts_to_sequences(sentences) # src_tokenizer에 등록된 index로 sentences를 index로 반환
y_train = tag_tokenizer.texts_to_sequences(pos_tags)  # tag_tokenizer에 등록된 index로 pos_tags를 index로 반환

print(X_train[:2])
print(y_train[:2])

max_len = 60 # 최대 길이는 97이지만 대부분의 문장들이 0~60 사이에 존재하므로 60으로 설정
X_train = pad_sequences(X_train, padding='post', maxlen=max_len) # max_len에 따라 부족하면 0으로 채우고, 넘치면 잘라 버림
y_train = pad_sequences(y_train, padding='post', maxlen=max_len) # max_len에 따라 부족하면 0으로 채우고, 넘치면 잘라 버림

# 8:2로 나누고 항상 동일한 결과를 얻기위해 random_state를 지정
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

type(X_train), type(y_train)

print('훈련 샘플 문장의 크기     : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기   : {}'.format(y_train.shape))
print('테스트 샘플 문장의 크기   : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))

import tensorflow as tf
from tensorflow.keras.models     import Sequential # from keras.models import Sequential
from tensorflow.keras.layers     import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

embedding_dim = 128
hidden_units  = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))     # vocab_size : 전체 문장에 포함된 단어의 갯수
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True))) 
model.add(TimeDistributed(Dense(tag_size, activation=('softmax')))) # tag_size : 태깅된 품사의 갯수

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()

# CPU 학습
print("CPU를 사용한 학습")
with tf.device("/device:CPU:0"): ### with tf.device("/device:GPU:0"):
    model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data=(X_test, y_test))
    print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

index_to_word = src_tokenizer.index_word
index_to_tag  = tag_tokenizer.index_word

i = 120 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측값 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1)      # 확률 벡터를 정수 레이블로 변환.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], y_test[i], y_predicted[0]):
    if word != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_tag[tag].upper(), index_to_tag[pred].upper()))
