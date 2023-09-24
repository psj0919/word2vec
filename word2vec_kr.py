import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import spacy
import torch

if __name__ == '__main__':
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="./data/ratings.txt")
    train_data = pd.read_table('./data/ratings.txt')
    print(train_data[:5])
    print(train_data.isnull().values.any())
    train_data = train_data.dropna(how='any')
    print(train_data.isnull().values.any())
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    nlp = spacy.load("xx_ent_wiki_sm")

    tokenized_data = []
    for sentence in tqdm(train_data['document']):
        doc = nlp(sentence)
        meaningful_words = [token.text for token in doc if token.text not in stopwords]
        tokenized_data.append(meaningful_words)

    print('리뷰의 최대 길이 :', max(len(review) for review in tokenized_data))
    print('리뷰의 평균 길이 :', sum(map(len, tokenized_data)) / len(tokenized_data))
    plt.hist([len(review) for review in tokenized_data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()



    model = Word2Vec(sentences=tokenized_data, vector_size=300, window=5, min_count=5, workers=4, sg=0)
    print(model.wv.most_similar("최민식"))
    print(model.wv.most_similar("히어로"))
