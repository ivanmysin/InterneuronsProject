import numpy as np
from params import rhythms_freqs_range
from scipy.signal import butter, filtfilt, hilbert

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def merge_ripple_zones(starts, ends, fs, gap_to_unite=5):
    """
    :param starts, ends: Массивы начал и окончаний рипплов
    :param fs: Частота дискретизации в Гц
    :param gap_to_unite: Максимальное время между рипплами, при котором они
                        считаются за один риппл в мс
    :return ripples: Массив. Первый индекс - начала риппла, второй индекс -
                     конец. Указано в единицах, которые подавались на вход
    """

    merge_end_indeces_to_delete = np.where(starts[1:] - ends[:-1] <= gap_to_unite * fs / 1000 )[0]
    merge_start_indeces_to_delete = merge_end_indeces_to_delete + 1
    ends = np.delete(ends, merge_end_indeces_to_delete)
    starts = np.delete(starts, merge_start_indeces_to_delete)

    ripples = np.vstack((starts, ends))
    return ripples

def get_ripples_episodes_indexes(filtered_lfp, fs,  ripple_frqs = rhythms_freqs_range['ripples']):

    """
    :param filtered_lfp: Сигнал лфп
    :params fs: Частота дискретизации
    :param ripple_frqs: Риппл частоты
    :return start_indxs, end_indxs: Массивы. Первый  - начала риппла, второй  -
                                    конец. Указано в единицах, которые подавались на вход
                                    (т.е. в частоте дискретизации)
    """
    envelope = np.abs(filtered_lfp)
    mean = np.mean(envelope)

    threshold = envelope - mean > 0

    ripples_for_start = np.concatenate((np.array([0], dtype=int), threshold)) # to account for boundary compications
    starts = ripples_for_start[:-1] < ripples_for_start[1:]
    start_indxs = np.where(starts == 1)[0]

    threshold = threshold[::-1]
    ripples_for_ends = np.concatenate((np.array([0], dtype=int), threshold)) # to account for boundary compications
    ends = ripples_for_ends[:-1] < ripples_for_ends[1:]
    ends = ends[::-1]
    end_indxs = np.where(ends == 1)[0]
    x = merge_ripple_zones(start_indxs, end_indxs, 10000)
    start_indxs, end_indxs = x[0] - 10/1000*fs, x[1] + 10/1000*fs

    ripples_epoches = np.vstack([start_indxs, end_indxs])
    return ripples_epoches

def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=10):
    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP
    :param theta_threshold : порог для отделения тета- от дельта-эпох
    :param accept_win : порог во времени, в котором переход не считается.
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    theta_lfp_abs = np.abs(theta_lfp)
    delta_lfp_abs = np.abs(delta_lfp)
    theta2delta = theta_lfp_abs / delta_lfp_abs
    theta2delta[theta2delta > 10] = 10  # Замена зашкаливающих значений
    theta2delta[np.isnan(theta2delta)] = 0
    is_theta = (theta2delta > theta_threshold).astype(np.int32)
    diff = np.diff(is_theta)
    diff = np.append(is_theta[0], diff)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))
    if start_idx[0] == 0:
        end_idx = np.append(end_idx, theta2delta.size - 1)

    accept_intervals = (end_idx - start_idx) > (accept_win*fs)
    start_idx = start_idx[accept_intervals]
    end_idx = end_idx[accept_intervals]
    theta_epoches = np.vstack([start_idx, end_idx])
    return theta_epoches



def get_circular_mean_R(filtered_lfp, spike_train):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """
    #fs - не нужно, т.к. спайки указаны в частоте записи лфп

    phase_signal = filtered_lfp
    y = np.take(phase_signal, spike_train)
    circular_mean = np.angle(np.mean(y)) + np.pi
    R = np.abs(np.mean(y))

    return circular_mean, R

def get_for_one_epoch(limits, spikes):
    x = spikes[(spikes >= limits[0]) & (spikes <= limits[1])].size
    return x

def get_mean_spike_rate_by_epoches(theta_epoches, non_theta_epoches, spike_train, fs):
    """
    :param theta_epoches: массив начал и концов тета эпох в формате
                          [[start, stop], [start, stop]]
    :param non_theta_epoches: массив начал и концов не-тета эпох в формате
                          [[start, stop], [start, stop]]
    :param spike_train: времена импульсов
    :param fs: частота дискретизации
    :return: среднее для тета эпох, ст.откл. для тета эпох,
             среднее для не-тета эпох, ст.откл. для не-тета эпох (дано в секундах)
    """

    theta = np.apply_along_axis(get_for_one_epoch, 1, theta_epoches, spike_train)
    non_theta = np.apply_along_axis(get_for_one_epoch, 1, non_theta_epoches, spike_train)
    theta = theta / fs
    non_theta = non_theta / fs

    theta_mean, theta_std = np.mean(theta), np.std(theta)
    non_theta_mean, non_theta_std = np.mean(non_theta), np.std(non_theta)

    return theta_mean, theta_std, non_theta_mean, non_theta_std


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




