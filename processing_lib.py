import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal
class Filtrate_lfp():
    def __init__(self, lowcut, highcut, fs, order):
        self.fs = fs
        self.nyq = self.fs * 0.5
        self.lowcut = lowcut / self.nyq
        self.highcut = highcut / self.nyq
        self.order = order
        self.b, self.a = signal.butter(N=self.order, Wn=[self.lowcut, self.highcut], btype='bandpass')

    def butter_bandpass_filter(self, lfp):
        filtered = signal.filtfilt(self.b, self.a, lfp)
        return filtered

def get_ripples_episodes_indexes(lfp, fs):
    """
    :param lfp: сигнал lfp
    :param fs: частота дискретизации
    :return:  массив начал и концов риппл событий
    """

    pass

def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=0.8):
    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP
    :param theta_threshold : порог для отделения тета- от дельта-эпох
    :param accept_win : порог во времени, в котором переход не считается.
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    theta_amplitude = np.abs(theta_lfp)
    delta_amplitude = np.abs(delta_lfp)
    
    relation = theta_amplitude / delta_amplitude
#     relation = relation[relation < 20]
    is_up_threshold = relation > theta_threshold
#     is_up_threshold = stats.zscore(relation) > theta_threshold 
# не забудь про изменение порога 
    is_up_threshold = is_up_threshold.astype(np.int32)

    f, a = plt.subplots(nrows=2, figsize=(15, 10))
    a[0].hist(theta_amplitude, bins=100)
    a[1].hist(delta_amplitude, bins=100)

    relation = relation[relation < 20]
    f, a = plt.subplots(nrows=1, figsize=(15, 5))
    a.hist(relation, bins=100)
    
    
    diff = np.diff(is_up_threshold)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))

    if start_idx[0] > end_idx[0]:
        start_idx = np.append(0, start_idx)

    if start_idx[-1] > end_idx[-1]:
        end_idx = np.append(end_idx, relation.size-1)
    
   
    # игнорируем небольшие пробелы между тета-эпохами
    large_intervals = (start_idx[1:] - end_idx[:-1]) > accept_win*fs
    large_intervals = np.append(True, large_intervals)
    start_idx = start_idx[large_intervals]
    end_idx = end_idx[large_intervals]

    # игнорируем небольшие тета-эпохи 
    large_th_epochs = (end_idx - start_idx) > accept_win*fs
    start_idx = start_idx[large_th_epochs]
    end_idx = end_idx[large_th_epochs] 

    # Все готово, упаковываем в один массив
    theta_epoches = np.append(start_idx, end_idx).reshape((2, start_idx.size))
    
    # Инвертируем тета-эпохи, чтобы получить дельта-эпохи
    non_theta_start_idx = end_idx[:-1]
    non_theta_end_idx = start_idx[1:]

    # Еще раз обрабатываем начало и конец сигнала
    if start_idx[0] > 0:
        non_theta_start_idx = np.append(0, non_theta_start_idx)
        non_theta_end_idx = np.append(start_idx[0], non_theta_end_idx)
    
    if end_idx[-1] < relation.size-1:
        non_theta_start_idx = np.append(non_theta_start_idx, end_idx[-1])
        non_theta_end_idx = np.append(non_theta_end_idx, relation.size-1)
    
    
    # Все готово, упаковываем в один массив
    non_theta_epoches = np.append(non_theta_start_idx, non_theta_end_idx).reshape((2, non_theta_start_idx.size))


    return theta_epoches, non_theta_epoches


def get_circular_mean_R(filtered_lfp, fs, spike_train):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param fs: частота дискретизации
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """

def get_mean_spike_rate_by_epoches(spike_train, epoches):
    """
    :param epoches: массив начал и концов эпох
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """
    pass


class InterneuronClassifier:

    def __int__(self, path2data):
        """
        :param path2data: Путь к файлу с данными, покоторым будем классифицировать
        :return:
        """

    def transpose(self, X):
        """
        :param X: вектор - описание потока импульсов
        :return: class name
        """




