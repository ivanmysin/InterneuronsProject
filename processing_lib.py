

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

def get_circular_mean_R(filtered_lfp, spike_train):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :return: циркулярное среднее и R
    """
    #fs - не нужно, т.к. спайки указаны в частоте записи лфп

    phase_signal = sig.hilbert(filtered_lfp)
    y = np.take(phase_signal, spike_train)
    circular_mean = np.angle(np.mean(y)) + np.pi
    R = np.abs(np.mean(y))

    return circular_mean, R

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




