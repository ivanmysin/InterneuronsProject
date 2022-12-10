

def get_ripples_episodes_indexes(lfp, fs):
    """
    :param lfp: сигнал lfp
    :param fs: частота дискретизации
    :return:  массив начал и концов риппл событий
    """

    pass

def get_theta_non_theta_epoches(theta_lfp, delta_lfp):
    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    pass

def get_circular_mean_R(filtered_lfp, fs, spike_train):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param fs: частота дискретизации
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """

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




