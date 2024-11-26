from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 使用sklearn自带的moon数据
X, y = make_moons(n_samples=100,noise=0.15,random_state=42)

# 绘制生成的数据
def plot_dataset(X,y,axis):
    plt.plot(X[:,0][y == 0],X[:,1][y == 0],'bs')
    plt.plot(X[:,0][y == 1],X[:,1][y == 1],'go')
    plt.axis(axis)
    plt.grid(True,which='both')


# 画出决策边界
def plot_pred(clf,axes):
    w = np.linspace(axes[0],axes[1], 100)
    h = np.linspace(axes[2],axes[3], 100)
    grid_x, grid_y = np.meshgrid(w, h)
    # grid_x 和 grid_y 被拉成一列，然后拼接成10000行2列的矩阵，表示所有点
    grid_xy = np.c_[grid_x.ravel(), grid_y.ravel()]
    # 二维点集才可以用来预测
    y_pred = clf.predict(grid_xy).reshape(grid_x.shape)
    # 等高线
    plt.contourf(grid_x, grid_y,y_pred,alpha=0.2)


ploy_kernel_svm_clf = Pipeline(
    steps=[
        ("scaler",StandardScaler()),
        ("svm_clf",SVC(kernel='poly', degree=3, coef0=1, C=5))
    ]
)


ploy_kernel_svm_clf.fit(X,y)

plot_pred(ploy_kernel_svm_clf,[-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
