import re
import nltk
import numpy as np
import scipy
from nltk.stem.porter import PorterStemmer
import csv

from sklearn import svm


# 邮件文本预处理
def vocaburary_mapping():
    vocab_list = {}
    with open('vocab.txt', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            vocab_list[row[1]] = int(row[0])
    return vocab_list


def email_preprocess(email):
    # 读取指定邮件文本
    with open(email, 'r') as f:
        email_contents = f.read()
    vocab_list = vocaburary_mapping()
    word_indices = []
    # 邮件文本预处理
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://\\S*', 'httpaddr', email_contents)
    email_contents = re.sub('\\S+@\\S+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%\n") + ']', email_contents)
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token.strip())
        if len(token) == 0:
            continue
        if token in vocab_list:
            word_indices.append(vocab_list[token])
    # 返回邮件文本词汇与词典中词汇的对应关系，以及得到的预处理文本
    return word_indices, ' '.join(tokens)


# 测试
word_indices, processed_contents = email_preprocess('emailSample1.txt')
print(word_indices)
print(processed_contents)


# 特征提取
def feature_extraction(word_indices):
    features = np.zeros((1899, 1))
    for index in word_indices:
        features[index] = 1
    return features


# 测试
features = feature_extraction(word_indices)
print(features)

# SVM分类
# 加载训练集
mat = scipy.io.loadmat("spamTrain.mat")
X, y = mat['X'], mat['y']

# 定义不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"\n使用核函数: {kernel}")
    # 训练SVM
    svm_model = svm.SVC(C=0.1, kernel=kernel)
    svm_model.fit(X, y.ravel())

    # 预测并计算训练集正确率
    predictions = svm_model.predict(X)
    predictions = predictions.reshape(np.shape(predictions)[0], 1)
    train_accuracy = (predictions == y).mean() * 100.0
    print(f'训练集正确率: {train_accuracy}%')

    # 加载测试集
    mat = scipy.io.loadmat("spamTest.mat")
    X_test, y_test = mat['Xtest'], mat['ytest']

    # 预测并计算测试集正确率
    predictions = svm_model.predict(X_test)
    predictions = predictions.reshape(np.shape(predictions)[0], 1)
    test_accuracy = (predictions == y_test).mean() * 100.0
    print(f'测试集正确率: {test_accuracy}%')

    # 查找权重最高词汇
    if kernel == 'linear':
        vocab_list = vocaburary_mapping()
        reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
        sorted_indices = np.argsort(svm_model.coef_, axis=None)
        print("权重最高的15个词汇:")
        for i in sorted_indices[0:15]:
            print(reversed_vocab_list[i])

    # 对样本进行预测
    word_indices, _ = email_preprocess('spamSample2.txt')
    features = feature_extraction(word_indices).transpose()
    print(f"对spamSample2.txt的预测结果: {svm_model.predict(features)}")
