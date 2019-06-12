"""

@file   : 002-pytorch实现cnn进行文本的分类.py

@author : xiaolu

@time   : 2019-06-11

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import os
import jieba
import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split


def is_chinese(uchar):
    # 判断一个unicode是否是一个汉字
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    # 判断一个unicode是否是数字
    if uchar >= '\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    # 判断一个unicode是否是英文字母
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_legal(uchar):
    # 判断是否为汉字，数字，和英文字符  也就是我们最后文本只要这些有用的字符，其他全部过滤掉
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True


def extract_chinese(line):
    # 过滤文本中的闲杂字符
    res = ""
    for word in line:
        if is_legal(word):
            res = res + word
    return res


def word2line(words):
    line = ''
    for word in words:
        line = line + ' ' + word
    return line


MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 20
VALIDATION_SPLIT = 0.2


# 数据预处理函数，在dir文件夹下每个子文件是一类内容  返回文本和标签
def datahelper(dir):
    labels_index = {}
    index_lables = {}
    num_recs = 0   # 统计有多少篇新闻

    fs = os.listdir(dir)    # 输出["搞笑", "科技", "教育"]
    # 标签整理
    i = 0
    for f in fs:
        labels_index[f] = i
        index_lables[i] = f
        i = i + 1
    print(labels_index)

    # 文本标签对应起来
    texts = []
    labels = []
    for la in labels_index.keys():     # 这里的key是文件夹名
        print(la + " " + index_lables[labels_index[la]])
        la_dir = dir + '/' + la    # dir是外层的目录
        fs = os.listdir(la_dir)
        for f in fs:
            file = open(la_dir + "/" + f, encoding='utf8')
            lines = file.readlines()
            for line in lines:
                if len(line) > 5:
                    line = extract_chinese(line)
                    words = jieba.lcut(line, cut_all=False)
                    text = words
                    texts.append(text)
                    labels.append(labels_index[la])
                    num_recs = num_recs + 1

    return texts, labels, labels_index, index_lables


# 加载词向量，可以事先预训练 采用gensim中word2vec模型训练
def getw2v():
    model_file_name = 'new_model_big.txt'
    # 模型训练，生成词向量

    # sentences = w2v.LineSentence('trainword.txt')    # trainword.txt是经过分词的文本
    # model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
    # model.save(model_file_name)    # 保存训练好的模型

    model = w2v.Word2Vec.load(model_file_name)    # 加载训练好的模型
    return model


train_dir = 'new_data'   # 存数据的那个目录
texts, labels, labels_index, index_lables = datahelper(train_dir)


# 定义textCNN模型
class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        vocab_size = args['vocab_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix = args['embedding_matrix']

        # 将实现训练好的词向量导入
        self.embeding = nn.Embedding(vocab_size, dim, _weight=embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # (16, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # (32, 30, 30)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # (64, 13, 13)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # (128, 9, 9)
        )

        self.out = nn.Linear(512, n_class)

    def forword(self, x):
        x = self.embeding(x)
        x = x.view(x.size(0), 1, max_len, word_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)   # 将(batch，outchanel,w,h)展平(batch，outchanel*w*h)
        output = self.out(x)
        return output


# 建立词表
word_vocab = []
word_vocab.append('')
for text in texts:    # texts结果 [[文本1分词结果], [文本2分词结果], [*]...]
    for word in text:
        word_vocab.append(word)


word_vocab = set(word_vocab)  # 去重
vocab_size = len(word_vocab)  # 去重后词的个数
# 设置词表的大小
nb_words = 40000
max_len = 64
word_dim = 20
n_class = len(index_lables)

args = {}
if nb_words < vocab_size:
    nb_words = vocab_size

args['vocab_size'] = nb_words
args['max_len'] = max_len
args['n_class'] = n_class
args['dim'] = word_dim


# 词到id的映射
word_to_idx = {word: i for i, word in enumerate(word_vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}


# 每个词对应的词向量  获取以训练好的词向量
embeddings_index = getw2v()
# 处理成能添加到embedding中的词向量
embeddings_matrix = np.zeros((nb_words, word_dim))
for word, i in word_to_idx.items():
    if i >= nb_words:
        continue
    if word in embeddings_index:    # 也就是说这个词我们给他进行了词嵌入
        embedding_vector = embeddings_index[word]   # 获取当前词的词向量
        if embedding_vector is not None:
            # 将词向量整个到矩阵中
            embeddings_matrix[i] = embedding_vector


args['embedding_matrix'] = torch.Tensor(embeddings_matrix)


# 构建textCNN模型
cnn = textCNN(args)


# 将文本转为id序列
texts_with_id = np.zeros([len(texts), max_len])
for i in range(0, len(texts)):
    if len(texts[i]) < max_len:    # 长度不够 有可能需要pad
        for j in range(0, len(texts[i])):
            texts_with_id[i][j] = word_to_idx[texts[i][j]]
        for j in range(len(texts[i]), max_len):    # 相当于pad   将长度不够的全部转为''
            texts_with_id[i][j] = word_to_idx['']
    else:
        for j in range(0, max_len):  # 长度够的话，直接截取max_len长
            texts_with_id[i][j] = word_to_idx[texts[i][j]]

# 定义优化器和损失函数
LR = 0.001
optimizer = optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

# 训练批次
batch_size = 10
texts_len = len(texts_with_id)
print(texts_len)

# 划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(texts_with_id, labels, test_size=0.2, random_state=42)


test_x = torch.LongTensor(x_test)
test_y = torch.LongTensor(y_test)
train_x = x_train
train_y = y_train
batch_test_size = 3

epoch = 10

for epoch in range(epoch):

    for i in range(0, len(train_x) // batch_size):

        b_x = Variable(torch.LongTensor(train_x[i*batch_size: i*batch_size + batch_size]))

        b_y = Variable(torch.LongTensor((train_y[i*batch_size: i*batch_size + batch_size])))
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(str(i))
        print(loss)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (b_y == pred_y)
        acc = acc.numpy().sum()
        accuracy = acc / (b_y.size(0))

    acc_all = 0
    for j in range(0, len(test_x) // batch_test_size):
        b_x = Variable(torch.LongTensor(test_x[j * batch_test_size: j * batch_test_size + batch_test_size]))
        b_y = Variable(torch.LongTensor((test_y[j * batch_test_size: j * batch_test_size + batch_test_size])))
        test_output = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # print(pred_y)
        # print(test_y)
        acc = (pred_y == b_y)
        acc = acc.numpy().sum()
        print("acc " + str(acc / b_y.size(0)))
        acc_all = acc_all + acc

    accuracy = acc_all / (test_y.size(0))
    print("epoch " + str(epoch) + " step " + str(i) + " " + "acc " + str(accuracy))

