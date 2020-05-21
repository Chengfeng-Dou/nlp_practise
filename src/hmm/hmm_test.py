import random

from src.hmm import HMM
import numpy as np


def init_s2s_s2o_pi():
    hmm = HMM()
    hmm.states = ['Sunny', 'Cloudy', 'Rainy']
    hmm.outputs = ['Dry', 'Dryish', 'Damp', 'Soggy']
    hmm.pi = np.array([0.63, 0.17, 0.20])
    hmm.s2s = np.array([
        [0.5, 0.375, 0.125],
        [0.25, 0.125, 0.625],
        [0.25, 0.375, 0.375]
    ])
    hmm.s2o = np.array([
        [0.6, 0.2, 0.15, 0.05],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.10, 0.35, 0.50]
    ])
    return hmm


def init_s2s_s2o_pi_2():
    hmm = HMM()
    hmm.states = ['rain', 'sun']
    hmm.outputs = ['walk', 'shop', 'clean']
    hmm.pi = np.array([0.6, 0.4])
    hmm.s2s = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    hmm.s2o = np.array([
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ])
    return hmm


def test_forward():
    hmm = init_s2s_s2o_pi()
    observe = ['Dry', 'Damp', 'Soggy']
    print(hmm.forward(observe))
    print(hmm.alpha)


def test_backward():
    hmm = init_s2s_s2o_pi()
    observe = ['Dry', 'Damp', 'Soggy']
    print(hmm.backward(observe))
    print(hmm.beta)


def test_viterbi():
    hmm = init_s2s_s2o_pi_2()
    observe = ['walk', 'shop', 'clean']
    print(hmm.viterbi(observe))


def test_em_step():
    hmm = init_s2s_s2o_pi()
    hmm.alpha = np.array([
        [0.378, 0.03031875, 0.00156859],
        [0.0425, 0.03770312, 0.00656563],
        [0.01, 0.02714688, 0.01876719]
    ])
    hmm.beta = np.array([
        [0.05984375, 0.18125, 1.],
        [0.0821875, 0.35625, 1.],
        [0.07875, 0.29375, 1.]
    ])
    observe = ['Dry', 'Damp', 'Soggy']  # 0, 2, 3
    xi, gamma = hmm.e_step(observe)
    print('xi\n', xi)
    print('gamma\n', gamma)
    delta = hmm.generate_I(observe)
    print('I\n', delta)
    hmm.m_step(delta, gamma, observe, xi)
    print('pi\n', hmm.pi)
    print('s2s\n', hmm.s2s)
    print('s2o\n', hmm.s2o)


def gen_data():
    a_b_init_critic = 0.2

    # state_change
    state_change = {
        "A": 0.3,  # 此时如果是A, 那么取random, 如果小于 此值就是A 否则为B
        "B": 0.6  # 此时如果是B, 那么取random, 如果小于 此值就是A 否则为B
    }
    # 点数情况
    # 点数对应的　index
    point = {"A": [0, 1, 2, 3, 4, 5], "B": [0, 1, 2, 3]}

    data_size = 100
    whole_data = []
    for i in range(data_size):
        dice = "A" if random.random() < a_b_init_critic else "B"
        data = []
        sequence_length = 10
        for _ in range(sequence_length):
            data.append(random.sample(point[dice], 1)[0])
            dice = "A" if random.random() < state_change[dice] else "B"

        whole_data.append(data)
    return whole_data
