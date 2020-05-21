from collections import Counter

import numpy as np


class MaxEnt:
    def __init__(self, learning_rate=0.1):
        self.feature_func = None  # 特征函数
        self.p_xy = None  # 联合概率密度
        self.p_x = None  # 边缘概率密度
        self.p_yx = None  # y 在条件 x 下的概率
        self.w = None  # 特征函数权重
        self.features = None  # 特征值
        self.labels = None  # 标签值
        self.n_sample = 0  # 样本数目
        self.dim = 0
        self.exp_feature = None  # 特征函数的经验期望
        self.lr = learning_rate

    def fit(self, X: np.ndarray, Y: np.ndarray, max_iter=100):
        """
        注意，X 和 Y 均为离散变量
        :param max_iter: 最大循环次数
        :param X:   特征
        :param Y:   标签
        :return:
        """
        self.init_param(X, Y)
        delta = np.ones([self.dim])
        i = 0
        while i < max_iter and np.sum(np.abs(delta) > 1):
            self.calculate_p_yx()
            for d in range(self.dim):
                delta[d] = self.lr * np.log(self.exp_feature[d] / self.est_fea(d))
            self.w += delta
            i += 1

    def init_param(self, X: np.ndarray, Y: np.ndarray):
        self.n_sample = Y.shape[0]
        self.dim = X.shape[1]  # X 的每一个维度都看成一个特征
        self.w = np.random.random([self.dim])

        # 用于记录每个维度当中 x 的种类
        self.features = [np.unique(X[:, d]).tolist() for d in range(self.dim)]
        # 用于按顺序记录标签
        self.labels = np.unique(Y).tolist()
        # 假设 X 每个维度的值不超过100种，标签不超过100种
        self.feature_func = np.zeros([self.dim, 100, 100])

        self.p_xy = Counter()
        self.p_x = Counter()
        self.p_yx = dict()
        # 统计每个维度上的特征于标签组对应的数目关系
        for x, y in zip(X, Y):
            x_index = [self.features[d].index(x[d]) for d in range(self.dim)]
            y_index = self.labels.index(y)

            for d in range(self.dim):  # 构造特征函数
                self.feature_func[d, x_index[d], y_index] = 1

            # 构造联合分布概率和边缘分布概率
            self.p_xy[tuple(x_index + [y_index])] += 1
            self.p_x[tuple(x_index)] += 1

        self.calculate_exp_feature()
        self.calculate_p_yx()

    def calculate_exp_feature(self):
        # 计算每个特征函数的经验分布
        self.exp_feature = np.zeros([self.dim])
        for xy_index in self.p_xy.keys():
            for d in range(self.dim):
                self.exp_feature[d] += self.pxy(xy_index) * self.f(xy_index, d)

    def f(self, xy_index, d):
        return self.feature_func[d, xy_index[d], xy_index[self.dim]]

    def calculate_p_yx(self):
        s = 0
        for xy_index in self.p_xy.keys():
            self.p_yx[xy_index] = self.py_in_condition_x(xy_index)
            s += self.p_yx[xy_index]

        for key in self.p_yx.keys():
            self.p_yx[key] /= s

    def py_in_condition_x(self, xy_index: tuple):
        # xy_index 最后一位是 y_index, 前面的是 x_index
        result = 0
        for d in range(self.dim):
            result += self.w[d] * self.f(xy_index, d)
        return np.exp(result)

    def px(self, x_index: tuple):
        return self.p_x[x_index] / self.n_sample

    def pxy(self, xy_index: tuple):
        return self.p_xy[xy_index] / self.n_sample

    def est_fea(self, d):
        """
        计算模型拟合的特征函数期望
        :param d: 第几个特征函数
        :return:
        """
        result = 0
        for xy_index in self.p_xy.keys():
            result += self.px(xy_index[:self.dim]) * self.p_yx[xy_index] * self.f(xy_index, d)

        return result

    def predict(self, X: np.ndarray):
        result = [self.predict_single(x) for x in X]
        return np.array(result)

    def predict_single(self, x):
        x_index = [self.features[d].index(x[d]) for d in range(self.dim)]
        p = [self.py_in_condition_x(tuple(x_index + [i])) for i in range(len(self.labels))]
        p = np.array(p)
        return self.labels[int(np.argmax(p))]
