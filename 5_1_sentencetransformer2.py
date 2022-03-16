# -*- coding: utf-8 -*-
"""5.1 SentenceTransformer2.ipynb
"""

! pip install -U sentence-transformers
import re
import sys
import nltk
import pandas as pd
import numpy  as np

from flask                           import Flask, render_template
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import euclidean_distances
from sentence_transformers           import SentenceTransformer
from nltk.corpus                     import stopwords
# nltk.download('stopwords')

# documents_df = pd.read_csv("/content/drive/MyDrive/data/test_label.csv", dtype={"id":object, "label" : object}, encoding='CP949')
# documents_df = pd.read_csv("Y:/SeeValue/ipynb/Python/Project/BigKinds/data/test_label.csv", 
#                           dtype={"id":object, "label" : object}, 
#                          encoding='CP949')

documents_df = pd.read_csv("/content/drive/MyDrive/data/test_label.csv", dtype={"id":object, "label" : object}, encoding='CP949')

from google.colab import drive
drive.mount('/content/drive')

sbert_model = SentenceTransformer('sentence-transformers/quora-distilbert-multilingual')

sbert_model

sentence_embeddings   = sbert_model.encode(documents_df['text'])

sentence_embeddings

def run_similar():
    sentence_embeddings   = sbert_model.encode(documents_df['text'])
    pairwise_similarities  = cosine_similarity(sentence_embeddings) # similarity_matrix 인수의 값으로 들어감
    
    return pairwise_similarities

def most_similar(doc_input, matrix): # 입력받은 문서와 비슷한 문서 찾기
    global documents_df
     
    # 새로운 문서 추가    
    doc_id       = documents_df['id'].astype(int).max()+1
    documents_df = documents_df.append({'id': str(doc_id), 'text' : doc_input}, ignore_index=True) # id와 새로운 문서 추가
    doc_ix       = documents_df.index[documents_df.id == str(doc_id)].tolist()[0]
    
    # 문서 유사도 계산    
    similarity_matrix = run_similar() 
    
    print (f'Document: {documents_df.iloc[doc_ix]["text"]}')
    print ('\n')
    print ('Similar Documents:')
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_ix])[::-1] # 상관계수 : 처음부터 끝까지 -1칸 간격으로 ( == 역순으로)
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_ix]) # 거리 : 오름차순 정열
        
    for ix in similar_ix:
        if ix == doc_ix:
            continue
        print('\n')
        print (f'Document: {documents_df.iloc[ix]["text"]}')
        print (f'{matrix} : {similarity_matrix[doc_ix][ix]}')

doc_input = '더불어민주당 의원들이 이재명 대선 후보의 아내 김혜경씨의 과잉 의전 논란, 법인카드 유용 의혹 관련 입장문을 페이스북에 공유했다가 삭제했다. ‘이재명 후보 선대위 공보단’ 명의의 입장문에는 김씨 의혹을 보도한 언론사를 향해 “오보로 판명될 때 보도에 책임을 져야 할 것”이라는 내용이 담겼다.이 후보의 최측근 그룹인 이른바 ‘7인회’ 의원 중 한 명인 김병욱 민주당 선대위 직능본부장과 이원욱 민주당 의원은 6일 밤 자신의 페이스북에 ‘김혜경씨에 SBS, KBS 보도에 대한 선대위 입장’이라는 제목의 글을 올렸다.해당 입장문에는 “김혜경씨에 대해 황제의전이 있었다는 보도로 사실 여부를 떠나 이미 김씨는 큰 상처를 입었다”라며 “SBS는 이 보도와 관련해 얼마나 사실 확인에 노력했는지 우선 묻는다”는 내용이 적혀 있었다. 또 제보자가 7급이 아닌 8급 별정직이라고 정정하기도 했다. 이어 “그는 이재명 지사가 취임할 때 성남시청에서 같이 온 배모 사무관(5급)이 데려와 채용한 경우이다”라며 “실제 김혜경씨는 A비서가 채용된 뒤 건강관리를 맡게 됐다며 인사 왔을 때 처음 본 이후 A비서를 따로 본 적이 전혀 없다. 단 한 번도 A비서에게 직접 일을 시킨 적 없다”고 했다.또 “반찬 조달, 음식 배달, 의약품 구매 등을 시켰다고 주장하고 있으나, 설혹 일부 그런 일이 있었다고 하더라도 이는 배 사무관의 지시였을 뿐이지 김씨는 관여하지도, 알지도 못하는 일”이라고도 했다.입장문에는 김씨가 A비서의 이름도 모른다며 “‘황제 의전’, ‘노예 생활’ 운운하는 것은 다분히 의도적이며 이치에 맞지 않다고 본다”라고도 적혀 있었다.'
tmp        = sys.stdout
sys.stdout = open('output.txt','w')
most_similar(doc_input, 'Cosine Similarity') # 문서 id 입력
sys.stdout.close()
sys.stdout = tmp 

with open('output.txt', 'r') as f:
    texts = f.read() # 파일을 통째로 읽기
    sim   = texts.split('\n')
    
    while '' in sim:    
        sim.remove('')
        
text_lst   = []
cosine_lst = []
for i, text in enumerate(sim):
    if i == len(sim)-1: 
        break
        
    if i == 0:
        text_lst.append(sim[i][10:])
        cosine_lst.append('1.0000000000000000')
    elif (i > 1) & (i % 2 == 0):
        text_lst.append(sim[i][10:])
        cosine_lst.append(sim[i+1][20:])
        
text_df       = pd.DataFrame(text_lst, columns=['Document'])
cosine_df     = pd.DataFrame(cosine_lst, columns=['Cosine Similarity'])
sim_documents = pd.concat([text_df, cosine_df], axis=1)

sim_documents.head()

sim_documents.tail()

sim_documents[sim_documents['Cosine Similarity'].astype(float) >= 0.93]

##### Flask PGM 예시
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form          = request.form
        result        = []
        bert_abstract = form['paragraph']
        question      = form['question']
        result.append(form['question'])
        result.append(answer_question(question, bert_abstract))
        result.append(form['paragraph'])
  
        return render_template("index.html", result=result)
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)









