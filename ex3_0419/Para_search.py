import scipy.io
import scipy.optimize
from sklearn import svm
from sklearn import model_selection
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集3 获得训练集和验证集
# 图像绘制函数
def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]
    # 正样本用+号绘制
    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    # 负样本用o 号绘制
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')

# 决策边界绘制函数
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
mat = scipy.io.loadmat("dataset_3.mat")
X, y = mat['X'], mat['y']  # 训练集
X_val, y_val = mat['Xval'], mat['yval']  # 验证集

# 绘制数据集3
plt.title('数据集3 分布')
plot(np.c_[X, y])
plt.show(block=True)

# 绘制验证集
plt.title('验证集分布')
plot(np.c_[X_val, y_val])
plt.show(block=True)

# 参数搜索
def params_search(X, y, X_val, y_val):
    c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    raveled_y = y.ravel()
    m_val = np.shape(X_val)[0]
    rbf_svm = svm.SVC(kernel='rbf')
    best = {'error': 999, 'C': 0.0, 'sigma': 0.0}
    for C in c_values:
        for sigma in sigma_values:
            # 根据不同参数训练 SVM
            rbf_svm.set_params(C=C)
            rbf_svm.set_params(gamma=1.0 / sigma)
            rbf_svm.fit(X, raveled_y)
            # 预测并计算误差
            predictions = []
            for i in range(0, m_val):
                prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
                predictions.append(prediction_result[0])
            predictions = np.array(predictions).reshape(m_val, 1)
            error = (predictions != y_val.reshape(m_val, 1)).mean()
            # 记录误差最小的一组参数
            if error < best['error']:
                best['error'] = error
                best['C'] = C
                best['sigma'] = sigma
    best['gamma'] = 1.0 / best['sigma']
    return best

# 训练高斯核函数SVM 并搜索使用最优模型参数
rbf_svm = svm.SVC(kernel='rbf')
best = params_search(X, y, X_val, y_val)
rbf_svm.set_params(C=best['C'])
rbf_svm.set_params(gamma=best['gamma'])
rbf_svm.fit(X, y)

# 绘制决策边界
plt.title('参数搜索后的决策边界')
plot(np.c_[X, y])
visualize_boundary(X, rbf_svm)
plt.show(block=True)