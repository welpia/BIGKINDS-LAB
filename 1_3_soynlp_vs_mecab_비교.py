# -*- coding: utf-8 -*-
"""1.3 soynlp vs Mecab 비교.ipynb
"""

!pip install soynlp

# 하나의 문서가 한 줄로 되어 있고 각 줄 내에서 문장은 두 개의 공백으로 분리되어 있는 형식의 말뭉치
!wget https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt -O /content/drive/MyDrive/data/2016-10-20.txt

import os
from google.colab import drive

drive.mount('/content/drive')
os.getcwd()

from google.colab import drive
drive.mount('/content/drive')

from soynlp import DoublespaceLineCorpus

corpus = DoublespaceLineCorpus("/content/drive/MyDrive/data/2016-10-20.txt") # 문서 단위 말뭉치 생성
len(corpus)                                                                  # 문서의 갯수

cd /content/drive/MyDrive/data



# 앞 5개의 문서 인쇄
i = 0
for d in corpus:
    print(i, d)
    i += 1
    if i > 4:
        break

corpus = DoublespaceLineCorpus("/content/drive/MyDrive/data/2016-10-20.txt", iter_sent=True) # 문장 단위 말뭉치 생성 
len(corpus)                                                                                  # 문장의 갯수

# 앞 5개의 문장 인쇄
i = 0
for d in corpus:
    print(i, d)
    i += 1
    if i > 4:
        break

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from soynlp.word import WordExtractor 
# 
# word_extractor = WordExtractor() # 단어 추출
# word_extractor.train(corpus)

word_score = word_extractor.extract() # extract() 메서드로 cohesion, branching entropy, accessor variety 등의 통계 수치를 계산

# Cohesion
print(word_score["연합"].cohesion_forward)
print(word_score["연합뉴"].cohesion_forward)
print(word_score["연합뉴스"].cohesion_forward)
print(word_score["연합뉴스는"].cohesion_forward)

# Branching Entropy
# 하나의 단어가 완결되는 위치에는 다양한 조사나 결합어가 올 수 있으므로 여러가지 글자의 확률이 비슷하게 나오고 따라서 엔트로피값이 높아진다.
print(word_score["연합"].right_branching_entropy)
print(word_score["연합뉴"].right_branching_entropy) # '연합뉴' 다음에는 항상 '스'만 나온다. 따라서 Branching Entropy = 0
print(word_score["연합뉴스"].right_branching_entropy)
print(word_score["연합뉴스는"].right_branching_entropy)

# Accessor Variety
# 특정 문자열 다음에 나올 수 있는 글자의 종류만 계산
print(word_score["연합"].right_accessor_variety)
print(word_score["연합뉴"].right_accessor_variety) # '연합뉴' 다음에는 항상 '스'만 나온다.
print(word_score["연합뉴스"].right_accessor_variety)
print(word_score["연합뉴스는"].right_accessor_variety)

"""## 토큰화"""

from soynlp.tokenizer import LTokenizer # 띄어쓰기가 잘 되어 있는 경우: L-토큰화

scores = {word:score.cohesion_forward for word, score in word_score.items()}
l_tokenizer = LTokenizer(scores=scores)

l_tokenizer.tokenize("안전성에 문제있는 스마트폰을 휴대하고 탑승할 경우에 압수한다", flatten=False)

from soynlp.tokenizer import MaxScoreTokenizer # 띄어쓰기가 안되어 있는 경우: Max Score 토큰화

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize("안전성에문제있는스마트폰을휴대하고탑승할경우에압수한다")

"""# Big Kinds"""

import pandas as pd

corona = pd.read_excel('/content/drive/MyDrive/data/corona.xlsx')
corona.columns

import re

corona['본문_1'] = corona['본문'].str.replace(pat=r'[^A-Za-z가-힣ㄱ-ㅎ]', repl=r' ', regex=True)

lines = corona['본문_1']

from soynlp.tokenizer import RegexTokenizer
from soynlp.noun      import LRNounExtractor

tokenizer = RegexTokenizer()
tokenizer

tokens = lines.apply(tokenizer.tokenize)

noun_extractor = LRNounExtractor(verbose=True)
noun_extractor.train(lines)

nouns = noun_extractor.extract()
nouns.items()

pd.DataFrame(nouns)

pd.DataFrame.from_dict(nouns).T

soynlp_nouns = pd.DataFrame.from_dict(nouns).T 
soynlp_nouns.to_csv('/content/drive/MyDrive/data/soynlp_nouns.csv')
