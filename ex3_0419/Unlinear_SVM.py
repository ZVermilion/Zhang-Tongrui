import scipy.io
import scipy.optimize
from sklearn import svm
from sklearn import model_selection
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

#图像绘制函数
def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]
    # 正样本用+号绘制
    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    # 负样本用o 号绘制
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')
#决策边界绘制函数
def visualize_boundary(X, trained_svm):
    kernel = trained_svm.get_params()['kernel']
    # 线性核函数
    if kernel == 'linear':
        w = trained_svm.coef_[0]
        i = trained_svm.intercept_
        xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        a = -w[0] / w[1]
        b = i[0] / w[1]
        yp = a * xp - b
        plt.plot(xp, yp, 'b-')
    # 高斯核函数
    elif kernel == 'rbf':
        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(X1))
        for i in range(0, np.shape(X1)[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)
        plt.contour(X1, X2, vals, colors='blue')

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-sum((x1 - x2) ** 2.0) / (2 * sigma ** 2.0))
# 计算高斯核函数
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
print("样本x1 和x2 之间的相似度: %f" % gaussian_kernel(x1, x2, sigma))
# 加载数据集2
mat = scipy.io.loadmat("dataset_2.mat")
X, y = mat['X'], mat['y']
# 绘制数据集2
plt.title('数据集2 分布')
plot(np.c_[X, y])
plt.show(block=True)
# 训练高斯核函数SVM
sigma = 0.01
rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma) # gamma 实际上是sigma 的倒数
rbf_svm.fit(X, y.ravel())
# 绘制非线性SVM 的决策边界
plt.title('高斯核函数SVM 决策边界')
plot(np.c_[X, y])
visualize_boundary(X, rbf_svm)
plt.show(block=True)
