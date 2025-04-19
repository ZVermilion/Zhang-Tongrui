import scipy.io
import scipy.optimize
from sklearn import svm
from sklearn import model_selection
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# 使 matplotlib 绘图支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]
    # 正样本用+号绘制
    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    # 负样本用o 号绘制
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')


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


# 加载数据集1
mat = scipy.io.loadmat("dataset_1.mat")
X, y = mat['X'], mat['y']

# 绘制数据集1
plt.title('数据集1 分布')
plot(np.c_[X, y])
plt.show()
# 训练线性SVM（C = 1）
linear_svm = svm.SVC(C=1, kernel='linear')
linear_svm.fit(X, y.ravel())
# 绘制 C=1 的 SVM 决策边界
plt.title('C=1 的SVM 决策边界')
plot(np.c_[X, y])
visualize_boundary(X, linear_svm)
plt.show(block=True)
# 训练线性SVM（C = 100）
linear_svm.set_params(C=100)
linear_svm.fit(X, y.ravel())
# 绘制 C=100 的SVM 决策边界
plt.title('C=100 的SVM 决策边界')
plot(np.c_[X, y])
visualize_boundary(X, linear_svm)
plt.show(block=True)