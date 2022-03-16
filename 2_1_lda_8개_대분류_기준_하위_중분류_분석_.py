# -*- coding: utf-8 -*-
"""2.1 LDA - 8개 대분류 기준 하위 중분류 분석 .ipynb
"""

import pickle
import numpy  as np
import pandas as pd

from gensim                       import corpora 
from gensim.models.callbacks      import CoherenceMetric  # 이 값이 작을수록 해당 토픽 모델은 실제 문헌 결과를 잘 반영한다는 뜻
from gensim.models.callbacks      import PerplexityMetric # 높을수록 의미론적 일관성 높음
from gensim.models.coherencemodel import CoherenceModel 
from gensim.models.ldamodel       import LdaModel 

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()

data     = 'Y:/SeeValue/ipynb/Python/Project/BigKinds/data'

len(document), document.columns

document.groupby(document['category']).count()

pd.crosstab(index=document['category'], columns=document['bigkinds'], margins=True, dropna=True)

def data_4_lda(category=1):
    document_cat = document[document.category == category]
    document_cat.reset_index(inplace=True, drop=True)
    
    processed_data = []
    
    for str in document_cat.bigkinds_nouns:
        if pd.isna(str) == False :
            processed_data.append(str.split(' '))
        elif pd.isna(str) == True :
            processed_data.append(['Nan'])      
            
    dictionary = corpora.Dictionary(processed_data)
    corpus     = [dictionary.doc2bow(text) for text in processed_data] # 데이터를 벡터화시키기
    
    return document_cat, dictionary, corpus

def run_lda(dictionary, corpus, category=1):
    if category == 1:   # 정치
        topics = 7
    elif category == 2: # 경제
        topics = 14
    elif category == 3: # 사회
        topics = 10
    elif category == 4: # 문화
        topics = 11
    elif category == 5: # 국제
        topics = 9
    elif category == 6: # 지역
        topics = 15
    elif category == 7: # 스포츠
        topics = 11
    elif category == 8: # IT_과학
        topics = 6
    elif category == 9: # 미분류
        topics = 1
        
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell') 
    coherence_logger  = CoherenceMetric(corpus=corpus, coherence="u_mass", logger='shell') 
    lda_model         = LdaModel(corpus, id2word=dictionary, num_topics=topics, random_state=1, passes=30, callbacks=[coherence_logger, perplexity_logger]) 
    topics            = lda_model.print_topics(num_words=5) 

    for topic in topics: 
        print(topic) 
        
    return lda_model

def save_model(category=1):
    pickle.dump(corpus, open('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/corpus.pkl', 'wb'))
    dictionary.save('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/dictionary.gensim') 
    lda_model.save('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/lda_model.gensim')
    
    vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False) 
    pyLDAvis.save_html(vis, 'Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/vis.html')

def load_model(category=1):
    dictionary = corpora.Dictionary.load('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/dictionary.gensim') 
    lda_model  = LdaModel.load('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/lda_model.gensim')
    print(lda_model.show_topic(0))
    return lda_model, dictionary

def apply_model(category=1):
    lst = []

    for i in range(len(corpus)):
        doc_topic_dist   = lda_model.get_document_topics(corpus[i], minimum_probability=0)
        sorted_doc_topic = sorted(doc_topic_dist, key=lambda x:x[1], reverse=True)
        lst.append(sorted_doc_topic[0])
        
    topics   = pd.DataFrame(lst, columns=['topic_id','prob'])
    df_topic = pd.concat([document_cat, topics], axis=1)        

    df_topic.to_excel('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/df_topic.xlsx')
        
    print(pd.crosstab(index=df_topic['bigkinds'], columns=df_topic['topic_id']))
    
    cross_table = pd.crosstab(index=df_topic['bigkinds'], columns=df_topic['topic_id'])
    cross_table.to_excel('Y:/SeeValue/ipynb/Python/Project/BigKinds/model/cat'+str(category)+'/cross_table.xlsx')
    
    return cross_table

for i in range(1,3): # range(1,8)을 사용하면 8개 주제를 동시에 실행 
    print('LDA 모델링을 위한 데이터 준비중', end='')
    document_cat, dictionary, corpus = data_4_lda(category=i)
    print('.......done!')
    print('LDA 모델링 실행', end='')
    lda_model = run_lda(dictionary, corpus, category=i)
    print('.......done!')
    print('LDA 모델 결과물 저장', end='')
    save_model(category=i)
    print('.......done!')
    print('LDA 모델 적용', end='')
    cross_table = apply_model(category=i)
    print('.......done!')
    print()

for i in range(3,8):
    print('LDA 모델링 Category = ', i)
    print('LDA 모델링을 위한 데이터 준비중', end='')
    document_cat, dictionary, corpus = data_4_lda(category=i)
    print('.......done!')
    print('LDA 모델링 실행', end='')
    lda_model = run_lda(dictionary, corpus, category=i)
    print('.......done!')
    print('LDA 모델 결과물 저장', end='')
    save_model(category=i)
    print('.......done!')
    print('LDA 모델 적용', end='')
    cross_table = apply_model(category=i)
    print('.......done!')
    print()



"""https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-2/

http://doc.mindscale.kr/km/unstructured/notebook/08_topic.html
"""

lda_model, dictionary = load_model(category=1)

data     = 'Y:/SeeValue/ipynb/data' # 실제 데이터 저장 장소로 변경
corona   = pd.read_excel(data+'/corona.xlsx', dtype={'뉴스식별자':str})
print('컬럼명(corona.xlsx) : ')
print(corona.columns, '\n')

len(corona), corona.columns

corona['본문'][0]

corona.groupby(corona['통합 분류1']).count()

corona['본문'] = corona['본문'].str.replace(pat=r'[^0-9A-Za-z가-힣ㄱ-ㅎ@ ]', repl=r'', regex=True)

from tqdm import tqdm

import konlpy
from konlpy.tag import Mecab

mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic/")

lines = corona['본문']

results   = []
words_all = []

print('[명사 추출 중입니다.]')
for line in tqdm(lines):
    r = []
    malist = mecab.nouns(line)
    for word in malist:
        if len(word) > 1 and word.isdigit() == False: # 두 글자 이상이면서 숫자가 아닌 경우만 저장
            r.append(word)
            words_all.append(word)
    r1 = (" ".join(r)).strip()
    results.append(r1)     
print('[명사 추출을 완료하였습니다.]\n')
print("추출된 전체 단어 수 : ", len(words_all))
print("추출된 단어 수      : ", len(set(words_all)))
print('\n')

processed_data = []
for str in results:
    if pd.isna(str) == False :
        processed_data.append(str.split(' '))
    elif pd.isna(str) == True :
        processed_data.append(['Nan'])

corpus = [dictionary.doc2bow(text) for text in processed_data] # 데이터를 벡터화시키기
len(corpus)

lst = []

for i in range(len(corpus)):
    doc_topic_dist   = lda_model.get_document_topics(corpus[i], minimum_probability=0)
    sorted_doc_topic = sorted(doc_topic_dist, key=lambda x:x[1], reverse=True)
    lst.append(sorted_doc_topic[0])
    
topics   = pd.DataFrame(lst, columns=['topic_id','prob'])
df_topic = pd.concat([corona, topics], axis=1)        
    
# pd.crosstab(index=df_topic['bigkinds'], columns=df_topic['topic_id'])

df_topic.head(3)
