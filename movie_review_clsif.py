import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import datasets
from torchtext.legacy.data import Field, BucketIterator
from model import GRU
import random

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1) # 레이블 값을 0(pos)과 1(neg)로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    predicted_labels = logit.max(1)[1].view(y.size()).data
    print("GT : {},".format(y.data), end='\n')
    print("PD : {}".format(predicted_labels.data))

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


if __name__=='__main__':
    SEED = 5
    random.seed(SEED)
    torch.manual_seed(SEED)

    gpu_id = "1"
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128 # batch_size
    lr = 0.001 # learning_rate
    EPOCHS = 100 # epoch
    # --------------------data_preprocessing -----------------------
    TEXT = Field(sequential=True, batch_first=True, lower=True)
    LABEL = Field(sequential=False, batch_first=True)
    #텐서 변환 및 데이터 유형을 정하기 위해 사용
    #sequential: 순서 반영, lower: 소문자화, batch_first: 배치를 제일먼저 출력소문자화
    # ---------------------------------------------------------

    # -------------data_load & validation ----------------
    trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
    #print(vars(trainset[0])) # text IMDB 리뷰에 해당하고 label 은 pos positive의 줄임말
    # -----------------------------------------------------


    # ------------- make_vocabulary ----------------------------
    TEXT.build_vocab(trainset, min_freq=5)  # 단어 집합 생성
    LABEL.build_vocab(trainset)

    vocab_size = len(TEXT.vocab)
    n_classes = 2
    #print('단어 집합의 크기 : {}'.format(vocab_size))
    #print('클래스의 개수 : {}'.format(n_classes))
    #print(TEXT.vocab.stoi)
    # ------------------------------------------------------------

    # -------------------------data_loader------------------------
    trainset, valset = trainset.split(split_ratio=0.8)  # train_data를 8:2로 분리해서 validataion_data를 만듬
    train_iter, val_iter, test_iter = BucketIterator.splits((trainset, valset, testset), batch_size=BATCH_SIZE, shuffle=True, repeat=False)
    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
    print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))

    # --------------------------------------------------------------

    # -----------------------train----------------------------------------------------
    model = GRU(1, 512, vocab_size,256, n_classes, 0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = None
    for e in range(1, EPOCHS+1):
        train(model, optimizer, train_iter)
        val_loss, val_accuracy = evaluate(model, val_iter)
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './checkpoints/txtclassification.pt')
            best_val_loss = val_loss


    model.load_state_dict(torch.load('./checkpoints/txtclassification.pt'))
    test_loss, test_acc = evaluate(model, test_iter)
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))