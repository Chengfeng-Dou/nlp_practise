import numpy as np

from src.wordcheck import WordsSM


class EDMatrix:
    def __init__(self, shape: list or tuple):
        self._matrix = np.zeros(shape)
        self._n_row = shape[0]
        self._n_col = shape[1]

    def __getitem__(self, index):
        return self._matrix[index]

    def _edit_distance(self, word1, i, word2, j):
        if word1[i] == word2[j]:  # 假如最后一个字符相同
            self._matrix[i, j] = self.value(i - 1, j - 1)

        elif word1[i] == word2[j - 1] and word1[i - 1] == word2[j]:
            # 假如倒数两字符交叉相同
            self._matrix[i, j] = 1 + min(
                self.value(i - 2, j - 2),  # 跳过倒数两字符，编辑距离 + 1
                self.value(i - 1, j),  # 可能是 word1 多了一个字符，跳过该字符
                self.value(i, j - 1)  # 可能是 word2 多了一个字符，跳过该字符
            )
        else:

            self._matrix[i, j] = 1 + min(
                self.value(i - 1, j - 1),
                self.value(i - 1, j),
                self.value(i, j - 1)
            )
        return self.value(i, j)

    def value(self, i, j):
        if i == -1:
            return j + 1
        if j == -1:
            return i + 1
        return self._matrix[i, j]

    def min_ed(self, word1: str, word2: str, l: int, u: int):
        """
        用于获取两字符串的最小编辑距离，该算法为动态规划算法
        :param word1: 正确的字符串，判断的长度为其本身的长度
        :param word2: 要检查的字符串，判断的长度为 [l, u）
        :param l: 开始下标
        :param u: 结束的下标 + 1
        :return: 最小编辑距离
        """
        min_ed = 100000
        # 继承父亲节点的运算结果
        self._matrix[len(word1) - 1, l - 1] = self.value(len(word1) - 2, l - 1)
        for i in range(l, len(word2)):
            ed = self._edit_distance(word1, len(word1) - 1, word2, i)
            if ed < min_ed and i < u:
                min_ed = ed
        return min_ed


class Checker:
    def __init__(self, sm: WordsSM, t=3):
        self._sm = sm
        self._stack = list()
        self._t = t

    def check(self, word: str):
        # 初始化栈
        self._stack.clear()
        self._stack.append(('', self._sm.root, 0))

        # 初始化输出
        result = list()

        # 初始化 H 矩阵
        ed_matrix = EDMatrix([30, len(word)])  # 这里行代表候选词，列表示 word

        while len(self._stack) != 0:
            # 弹出上一个节点
            # string 到该节点为止字符串，state 表示该节点，i 表示本轮该访问的子结点的 id
            string, state, i = self._stack.pop()
            # 如果该节点的所有子结点已经遍历完成，那么直接跳过
            if state.next is None or i >= len(state.next):
                continue
            # 获取当前状态，并获取新的字符串
            cur_state = state.next[i]
            cur_string = string + cur_state.value
            # 将下一个要遍历的兄弟节点压入栈中
            self._stack.append((string, state, i + 1))
            # 获取该字符串与目标字符串的最小编辑距离，保证两者的编辑距离不会过大，用于剪枝
            min_ed = ed_matrix.min_ed(cur_string, word,
                                      max(0, len(cur_string) - self._t - 1),
                                      min(len(word), len(cur_string) + self._t))
            if min_ed <= self._t:
                # 深度优先遍历
                self._stack.append((cur_string, cur_state, 0))

            ed = ed_matrix.value(len(cur_string) - 1, len(word) - 1)
            if cur_state.is_final and ed <= self._t:
                result.append((cur_string, ed))

        # 显示最有可能的前10个单词
        result = sorted(result, key=lambda x: x[1])[:10]
        return [w[0] for w in result]
