import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import torch
nltk.download('punkt')

from gensim.models import Word2Vec
from gensim.models import KeyedVectors


if __name__ == '__main__':
    gpu_id = '0'
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    # download_data
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
        filename="./data/ted_en-20160408.xml")

    targetXML = open('./data/ted_en-20160408.xml', 'r', encoding='UTF8')
    target_text = etree.parse(targetXML)
    parse_text = '\n'.join(target_text.xpath('//content/text()'))
    content_text = re.sub(r'\([^)]*\)', '', parse_text)
    sent_text = sent_tokenize(content_text).to(device)
    normalized_text = []
    for string in sent_text:
        tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
        normalized_text.append(tokens)

    result = [word_tokenize(sentence) for sentence in normalized_text]
    print('총 샘플의 개수 : {}'.format(len(result)))

    model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
    model_result = model.wv.most_similar("man")
    print(model_result)

    model.wv.save_word2vec_format('eng_w2v')  # 모델 저장
    loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")  # 모델 로드

    model_result = loaded_model.most_similar("man")
    print(model_result)

