import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import parzen
from numba import jit

@jit(nopython=True)
def __clear_articacts(lfp, win, threshold):
    lfp = lfp - np.mean(lfp)
    lfp_std = np.std(lfp)
    is_large = np.logical_and( (lfp > 10*lfp_std), (lfp < 10*lfp_std) )
    is_large = is_large.astype(np.float64)
    is_large = np.convolve(is_large, win)
    is_large = is_large[win.size // 2:-win.size // 2 + 1]
    is_large = is_large > threshold
    lfp[is_large] = np.random.normal(0, 0.001*lfp_std, np.sum(is_large) )
    return lfp

def clear_articacts(lfp, win_size=101, threshold=0.1):
    win = parzen(win_size)
    lfp = __clear_articacts(lfp, win, threshold)
    return lfp

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


@jit(nopython=True)
def get_ripples_episodes_indexes(ripples_lfp, fs, threshold=4, accept_win=0.02):
    """
    :param ripples_lfp: сигнал lfp, отфильтрованный в риппл-диапазоне
    :param fs: частота дискретизации
    :param threshold: порог для определения риппла
    :param accept_win: минимальная длина риппла в сек
    :return:  списки начал и концов риппл событий в единицах, указанных в частоте дискретизации (fs)
    """

    ripples_lfp_th = threshold * np.std(ripples_lfp.real)
    ripples_abs = np.abs(ripples_lfp)
    is_up_threshold = ripples_abs > ripples_lfp_th
    is_up_threshold = is_up_threshold.astype(np.int32)
    diff = np.diff(is_up_threshold)
    diff = np.append(is_up_threshold[0], diff)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))

    if start_idx[0] > end_idx[0]:
        end_idx = end_idx[1:]

    if start_idx[-1] > end_idx[-1]:
        start_idx = start_idx[:-1]

    accept_intervals = (end_idx - start_idx) > accept_win * fs
    start_idx = start_idx[accept_intervals]
    end_idx = end_idx[accept_intervals]

    ripples_epoches = np.append(start_idx, end_idx).reshape((2, start_idx.size))
    return ripples_epoches

@jit(nopython=True)
def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=2):
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
    is_up_threshold = relation > theta_threshold
    is_up_threshold = is_up_threshold.astype(np.int32)
    diff = np.diff(is_up_threshold)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))

    if start_idx[0] > end_idx[0]:
        start_idx = np.append(0, start_idx)

    if start_idx[-1] > end_idx[-1]:
        end_idx = np.append(relation.size-1, end_idx)

    # удаляем небольшие пробелы между тета-эпохами
    is_large_intervals = (end_idx[:-1] - start_idx[1:])*fs > accept_win
    is_large_intervals = np.append(True, is_large_intervals)
    start_idx = start_idx[is_large_intervals]
    end_idx = end_idx[is_large_intervals]

    # удаляем небольшие тета-эпохи меньще порога
    is_large_intervals = (end_idx - start_idx)*fs > accept_win
    is_large_intervals = np.append(True, is_large_intervals)
    start_idx = start_idx[is_large_intervals]
    end_idx = end_idx[is_large_intervals]

    # Все готово, упаковываем в один массив
    theta_epoches = np.append(start_idx, end_idx).reshape((2, start_idx.size))

    # Инвертируем тета-эпохи, чтобы получить дельта-эпохи
    non_theta_start_idx = end_idx[:-1]
    non_theta_end_idx = start_idx[1:]

    # Еще раз обрабатываем начало и конец сигнала
    if start_idx[0] != 0:
        non_theta_start_idx = np.append(0, non_theta_start_idx)
        non_theta_end_idx = np.append(start_idx[0], non_theta_end_idx)

    if end_idx[-1] != relation.size-1:
        non_theta_start_idx = np.append(end_idx[-1], non_theta_start_idx)
        non_theta_end_idx = np.append(relation.size-1, non_theta_end_idx)

    # Все готово, упаковываем в один массив
    non_theta_epoches = np.append(non_theta_start_idx, non_theta_end_idx).reshape((2, non_theta_start_idx.size))


    return theta_epoches, non_theta_epoches


@jit(nopython=True)
def get_circular_mean_R(filtered_lfp, spike_train, mean_calculation = 'uniform'):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :mean_calculation: способ вычисления циркулярного среднего и R
    :return: циркулярное среднее и R
    """
    #fs - не нужно, т.к. спайки указаны в частоте записи лфп
    if spike_train.size == 0:
        return np.nan, np.nan
    if mean_calculation == 'uniform':
        angles = np.angle(np.take(filtered_lfp, spike_train))
        mean = np.mean(np.exp(angles * 1j))
        circular_mean = np.angle(mean)
        R = np.abs(mean)
        return circular_mean, R

    elif mean_calculation == 'normalized':
        phase_signal = np.take(filtered_lfp, spike_train)
        phase_signal = phase_signal / np.sum(np.abs(phase_signal))
        mean = np.sum(phase_signal)
        circular_mean = np.angle(mean)
        R = np.abs(mean)
        return circular_mean, R
    else:
        raise ValueError("This mean_calculation is not acceptable")
@jit(nopython=True)
def get_for_one_epoch(limits, spikes):
    x = spikes[(spikes >= limits[0]) & (spikes < limits[1])]
    return x
@jit(nopython=True)
def get_over_all_epoches(epoches_indexes, spike_train):
    spikes_during_epoches = np.empty(0, dtype=spike_train.dtype)
    for (start_idx, end_idx) in epoches_indexes:
        spikes_in_epoch = get_for_one_epoch((start_idx, end_idx), spike_train)
        spikes_during_epoches = np.append(spikes_during_epoches, spikes_in_epoch)
    return spikes_during_epoches

@jit(nopython=True)
def get_mean_spike_rate_by_epoches(epoches_indexes, spike_train, samplingRate):
    """
    :param epoches_indexes: массив начал и концов тета эпох в формате
                          [[start, stop], [start, stop]]
    :param spike_train: индексы импульсов
    :param samplingRate: частота дискретизации
    :return: среднее для  эпох, ст.откл.
    """

    spikes = []
    for (start_idx, end_idx) in epoches_indexes:
        spikes_in_epoches = get_for_one_epoch((start_idx, end_idx), spike_train)
        spikes_rate = spikes_in_epoches.size / (end_idx - start_idx) * samplingRate
        spikes.append(spikes_rate)
    spikes = np.asarray(spikes)
    if spikes.size == 0:
        return 0, 0
    spike_rate = np.mean(spikes)
    spike_rate_std = np.std(spikes)

    return spike_rate, spike_rate_std


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




