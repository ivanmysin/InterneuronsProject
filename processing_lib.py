import numpy as np
from params import rhythms_freqs_range
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import parzen


def clear_articacts(lfp, win_size=101, threshold=0.1):
    lfp = lfp - np.mean(lfp)
    lfp_std = np.std(lfp)
    is_large = np.logical_and( (lfp > 10*lfp_std), (lfp < 10*lfp_std) )
    is_large = is_large.astype(np.float64)
    is_large = np.convolve(is_large, parzen(win_size), mode='same')
    is_large = is_large > threshold

    lfp[is_large] = np.random.normal(0, 0.001*lfp_std, np.sum(is_large) )
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

def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=2):
    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP
    :param theta_threshold : порог для отделения тета- от дельта-эпох
    :param accept_win : порог во времени, в котором переход не считается.
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    sampling_period = 1 / fs

    theta_amplitude = np.abs(theta_lfp)
    delta_amplitude = np.abs(delta_lfp)

    relation = theta_amplitude / delta_amplitude

    theta_state_inds = []
    non_theta_state_inds = []

    for ind, i in enumerate(relation):
        if i >= theta_threshold:
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

        if secs >= accept_win:
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

        if secs >= accept_win:
            time_filtered_non_theta_states.append(state)

    theta_epoches = np.asarray(time_filtered_theta_states)
    non_theta_epoches = np.asarray(time_filtered_non_theta_states)
    return theta_epoches, non_theta_epoches



def get_circular_mean_R(filtered_lfp, spike_train):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """
    #fs - не нужно, т.к. спайки указаны в частоте записи лфп

    phase_signal = filtered_lfp
    y = np.take(phase_signal, spike_train)
    mean_y = np.mean(y)
    circular_mean = np.angle(mean_y) #+ np.pi
    R = np.abs(mean_y)

    return circular_mean, R

def get_for_one_epoch(limits, spikes):
    x = spikes[(spikes >= limits[0]) & (spikes < limits[1])]
    return x
def get_over_all_epoches(epoches_indexes, spike_train):
    spikes_during_epoches = np.empty(0, dtype=spike_train.dtype)
    for (start_idx, end_idx) in epoches_indexes:
        spikes_in_epoch = get_for_one_epoch((start_idx, end_idx), spike_train)
        spikes_during_epoches = np.append(spikes_during_epoches, spikes_in_epoch)
    return spikes_during_epoches

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




