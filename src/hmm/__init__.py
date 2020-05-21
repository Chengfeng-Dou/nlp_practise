import numpy as np


def init_p_array(n_row, n_col):
    arr = np.random.rand(n_row, n_col)
    row_sum = np.sum(arr, axis=1)[:, None]
    return arr / row_sum


def init_p_vector(length):
    vec = np.random.rand(length)
    return vec / np.sum(vec)


class HMM(object):
    def __init__(self):
        self.states = None  # 状态集
        self.outputs = None  # 输出集
        self.s2s = None  # 状态转移矩阵
        self.s2o = None  # 状态输出矩阵
        self.pi = None  # 初始分布矩阵

        self.trained = False  # 模型是否训练过

        self.alpha = None  # 前向传播变量矩阵
        self.beta = None  # 后向传播变量矩阵

        self.delta = None  # 维特比算法概率矩阵
        self.phi = None  # 维特比算法路径矩阵

    def decode(self, sequence: list or tuple or str):
        if not self.trained:
            raise Exception('model has not initialized!')
        return self.forward(sequence)

    def forward(self, observe: list or tuple or str):
        # 行代表状态，列代表时序
        T = len(observe)
        self.alpha = np.zeros([len(self.states), T])
        self.alpha[:, 0] = self.s2o[:, self.outputs.index(observe[0])] * self.pi
        # 动态规划 alpha[i, t] 表示在第 t 时刻状态为 i，且观测到 observes[: t + 1] 的概率
        for t in range(1, T):
            self.alpha[:, t] = (np.dot(self.alpha[:, t - 1], self.s2s)
                                * self.s2o[:, self.outputs.index(observe[t])])
        # 最后的概率为最后一列相加，表示观测到 observes 且最后时刻状态为任意状态的概率
        return np.sum(self.alpha[:, T - 1])

    def backward(self, observe: list or tuple or str):
        T = len(observe)
        self.beta = np.zeros([len(self.states), T])
        self.beta[:, T - 1] = 1

        for t in reversed(range(T - 1)):
            self.beta[:, t] = (
                np.dot(
                    self.beta[:, t + 1] * self.s2o[:, self.outputs.index(observe[t + 1])],
                    self.s2s.T)
            )

        return np.sum(self.pi * self.s2o[:, self.outputs.index(observe[0])] * self.beta[:, 0])

    def viterbi(self, observe: list or tuple or str):
        if not self.trained:
            raise Exception('model has not initialized!')
        T = len(observe)
        self.delta = np.zeros([len(self.states), T])
        self.phi = np.zeros([len(self.states), T])

        self.delta[:, 0] = self.pi * self.s2o[:, self.outputs.index(observe[0])]
        self.phi[:, 0] = -1

        for t in range(1, T):
            prob = self.delta[:, t - 1].reshape([-1, 1]) * self.s2s
            index = np.argmax(prob, axis=0)
            ot = self.outputs.index(observe[t])
            for j in range(len(self.states)):
                self.delta[j, t] = prob[index[j], j] * self.s2o[j, ot]
                self.phi[j, t] = index[j]

        final_state = np.argmax(self.delta[:, T - 1])
        max_prob = self.delta[final_state, T - 1]

        result = [self.states[final_state]]
        cur_state = final_state

        for i in reversed(range(T)):
            cur_state = int(self.phi[cur_state, i])
            if cur_state != -1:
                result.insert(0, self.states[cur_state])

        return max_prob, result

    def m_step(self, I, gamma, xi, observe):
        T = len(observe)
        self.pi = gamma[0, :]
        self.s2s = np.sum(xi, axis=0) / np.sum(gamma[: T - 1], axis=0)
        self.s2o = (np.dot(I, gamma) / np.sum(gamma, axis=0)).T

    def e_step(self, observe: list or tuple or str):
        # 首先计算 alpha 和 beta 的值
        self.forward(observe)
        self.backward(observe)
        # xi[t, i, j] 代表在第 t 时刻状态为 i，在 t+1 时刻状态为 j 的概率
        # 状态转移概率只有 T - 1 个时刻有
        T = len(observe)
        xi = np.zeros([T - 1, len(self.states), len(self.states)])

        for t in range(T - 1):
            o = self.outputs.index(observe[t + 1])
            xi[t, :, :] = (
                    self.alpha[:, t, None] *
                    (self.beta[:, t + 1] * self.s2o[:, o]) *
                    self.s2s
            )

            xi[t, :, :] /= xi[t, :, :].sum()

        # gamma[t, i] 表示第 t时刻状态为 i 的概率，一共有 T 个
        tmp = self.alpha * self.beta  # 这里 tmp[i, j] 表示在 j 时刻状态为 i 的非标准化概率
        gamma = (tmp / np.sum(tmp, axis=0)).T
        return xi, gamma

    def generate_I(self, observe: list or tuple or str):
        T = len(observe)
        om = np.zeros([len(self.outputs), T])
        for t in range(T):
            om[self.outputs.index(observe[t]), t] = 1
        return om

    def baum_welch(self, observe: str or list or tuple, max_iter=100):
        self.s2s = init_p_array(len(self.states), len(self.states))
        self.s2o = init_p_array(len(self.states), len(self.outputs))
        self.pi = init_p_vector(len(self.states))

        I = self.generate_I(observe)

        x, lamb, itr = 1, 2, 0
        while itr < max_iter and lamb > x:
            memory = [self.s2s, self.s2o, self.pi]

            xi, gamma = self.e_step(observe)
            self.m_step(I, gamma, xi, observe)

            delta_s2s = np.abs(memory[0] - self.s2s)
            delta_s2o = np.abs(memory[1] - self.s2o)
            delta_pi = np.abs(memory[3] - self.pi)

            lamb = np.sum(delta_s2s) + np.sum(delta_s2o) + np.sum(delta_pi)
            itr += 1

        return self.s2s, self.pi, self.s2o
