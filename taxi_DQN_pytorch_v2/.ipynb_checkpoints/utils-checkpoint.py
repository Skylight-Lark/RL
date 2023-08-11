import numpy as np


def moving_average(array, window_size):
    cumulative_sum = np.cumsum(np.insert(array, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size, 2)  # 调整 r 数组的长度
    begin = np.cumsum(array[:window_size-1])[::2] / r
    end = (np.cumsum(array[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def encode_state(state):
    ''' 将状态编码为one-hot向量 '''
    one_hot = np.zeros(500)
    one_hot[state] = 1.0
    return one_hot