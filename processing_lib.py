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

def get_theta_non_theta_epoches(theta_lfp, delta_lfp, sampling_period):

    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    theta_analytic_signal = signal.hilbert(theta_lfp)
    delta_analytic_signal = signal.hilbert(delta_lfp)

    theta_analytic_signal = theta_analytic_signal.reshape((-1, 1))
    delta_analytic_signal = delta_analytic_signal.reshape((-1, 1))

    theta_amplitude = np.abs(theta_analytic_signal)
    delta_amplitude = np.abs(delta_analytic_signal)

    relation = theta_amplitude / delta_amplitude

    theta_state_inds = []
    non_theta_state_inds = []

    for ind, i in enumerate(relation):
        if i >= 2:
            theta_state_inds.append(ind)
        else:
            non_theta_state_inds.append(ind)

    theta_state_inds = np.asarray(theta_state_inds)
    theta_array = theta_state_inds[1:] - theta_state_inds[:-1]

    non_theta_state_inds = np.asarray(non_theta_state_inds)
    non_theta_array = non_theta_state_inds[1:] - non_theta_state_inds[:-1]

    theta_states = []
    start_theta = 0

    for i in range(len(theta_array)):
        if theta_array[i] != 1:
            stop_theta_ind = theta_state_inds[i]

            epoch = [theta_state_inds[start_theta], stop_theta_ind]

            theta_states.append(epoch)

            start_theta = i + 1

    time_filtered_theta_states = []

    for state in theta_states:

        length = state[1] - state[0]
        secs = length / sampling_period

        if secs >= 7:
            time_filtered_theta_states.append(state)

    non_theta_states = []
    start_non_theta = 0

    for i in range(len(non_theta_array)):
        if non_theta_array[i] != 1:
            stop_non_theta = non_theta_state_inds[i]

            epoch = [non_theta_state_inds[start_non_theta], stop_non_theta]

            non_theta_states.append(epoch)

            start_non_theta = i + 1

    time_filtered_non_theta_states = []

    for state in non_theta_states:

        length = state[1] - state[0]
        secs = length / sampling_period

        if secs >= 3:
            time_filtered_non_theta_states.append(state)

    theta_nontheta_dict = {'theta_state': time_filtered_theta_states,
                           'non_theta_state': time_filtered_non_theta_states}
    return (theta_nontheta_dict)

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




