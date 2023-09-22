import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import tqdm

if __name__ == '__main__':
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="./data/ratings.txt")
    train_data = pd.read_table('./data/ratings.txt')
    print(train_data[:5])
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()

    tokenized_data = []
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        tokenized_data.append(stopwords_removed_sentence)

    print('리뷰의 최대 길이 :', max(len(review) for review in tokenized_data))
    print('리뷰의 평균 길이 :', sum(map(len, tokenized_data)) / len(tokenized_data))
    plt.hist([len(review) for review in tokenized_data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()

    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)
    print(model.wv.most_similar("최민식"))
    print(model.wv.most_similar("히어로"))
