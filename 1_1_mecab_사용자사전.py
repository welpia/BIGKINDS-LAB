# -*- coding: utf-8 -*-
"""1.1 Mecab 사용자사전.ipynb

"""

import pandas as pd
from tqdm import tqdm

data = 'Y:/SeeValue/ipynb/Python/Project/BigKinds/data'

larm_noun_dic              = pd.read_table(data+'/larm_noun_dic.txt', encoding='cp949')
larm_noun_dic['LARM_NOUN'] = larm_noun_dic['LARM_NOUN'].str.replace(' ', '')
len(larm_noun_dic)

larm_words = list(set(larm_noun_dic['LARM_NOUN']))
len(larm_words)

bigkinds_words = []
for word in tqdm(larm_words):
    if (word in mecab_words) == False:
        bigkinds_words.append(word)

len(bigkinds_words)

from jamo import h2j, j2hcj 

JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def jongsung_check(word):
    jamo_str = j2hcj(h2j(word[len(word)-1])) 
  
    return jamo_str[len(jamo_str)-1] in JONGSUNG_LIST

custom = []
for noun in bigkinds_words:
    if jongsung_check(noun) == True: # 종성이 있는 경우
        jongsung_yn = 'T'
    elif jongsung_check(noun) == False: # 종성이 없는 경우
        jongsung_yn = 'F'
    custom.append(noun+',,,,NNP,*,'+jongsung_yn+','+noun+',*,*,*,*,*\n')

len(custom)

with open("C:/mecab/user-dic/custom.csv", 'w', encoding='utf-8') as f: 
    for line in custom: 
        f.write(line)

"""### BigKinds 사용자 사전 등록"""

with open("C:/mecab/user-dic/nnp.csv", 'r', encoding='utf-8') as f: 
    file_data = f.readlines()
file_data

larm_nnp = []
for noun in larm_noun_dic.LARM_NOUN:
    if jongsung_check(noun) == True:
        jongsung_yn = 'T'
    elif jongsung_check(noun) == False:
        jongsung_yn = 'F'
    larm_nnp.append(noun+',,,,NNP,*,'+jongsung_yn+','+noun+',*,*,*,*,*\n')
    file_data.append(noun+',,,,NNP,*,'+jongsung_yn+','+noun+',*,*,*,*,*\n')

with open("C:/mecab/user-dic/custom.csv", 'w', encoding='utf-8') as f: 
    for line in file_data: 
        f.write(line)

with open("C:/mecab/user-dic/custom.csv", 'r', encoding='utf-8') as f: 
    file_data = f.readlines()
len(file_data)

"""### Windows에서는 Powershell에서 관리자 권한으로 실행해야 하는데 linux에서는 잘 모르겠음.
- mecab이 import된 상태이면 프로세스가 잡혀있어서 오류발생.

PS C:\WINDOWS\mecab> .\tools\add-userdic-win.ps1
"""

from konlpy.tag import Mecab

mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic/")

mecab.morphs('나는 1968레트로스탠드에 다닙니다')

mecab.morphs('스웨커페이스옵티마이저')

"""### BigKinds 사용자 사전 우선순위 조정"""

with open("C:/mecab/mecab-ko-dic/user-custom.csv", 'r', encoding='utf-8') as f: 
    file_data = f.readlines()
len(file_data)

file_data[0:10]

file_data0 = []
for text in file_data:
    str    = text.split(',')
    str[3] = '0'
    nnp    = str[0]
    for j in range(1,len(str)):
        nnp = nnp + ',' + str[j]
    file_data0.append(nnp)
len(file_data0)

file_data0[0:10]
