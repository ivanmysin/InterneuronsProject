import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import parzen
from numba import jit

#@jit(nopython=True)
def clear_articacts(lfp, fs = 1250, win_len = 2500, tol_thresh_mult = 15, min_len_mult = 3, show = False):
    #win_len - длительнгость фрагментов, на которые нарезается сигнал
    #tol_thresh_mult - чистка проводится по выбросам (fliers), но многие выбросы визуально близки
    #к "нормальным" значениям размаха потенциала, из-за чего сигнал слишком мелко нарезается, если
    #учитывать все выбросы. Какую-то долю выбросов можно перевести в категорию "нормальных" значений,
    #что и регулируется этим параметром.
    #min_len_mult - для того, чтобы убрать самые короткие фрагменты, например, если фрагмент 
    #чистого сигнала меньше трех длительностей минимального фрагмента, то он отбрасывается.
    mm_list = [] #список размаха значений (разница между минимумом и максимумом) фрагментов lfp
    times = np.arange(len(lfp)) / 1250
    for i in range(0, int(np.floor(len(lfp)/win_len))):
        lfp_fragment = list(lfp[i * win_len : (i + 1) * win_len])
        mm_list.append(abs(np.max(lfp_fragment) - np.min(lfp_fragment)) )
    b = plt.boxplot(mm_list)
    plt.close()
    for i in b["fliers"]:
        mm_fliers = i.get_data()[1]
    fl_idxs = [mm_list.index(i) for i in mm_fliers]
    
    anti_fl_idxs = [i for i in range(len(mm_list)) if i not in fl_idxs]
    tol_thresh = np.std(np.asarray(mm_list)[anti_fl_idxs]) * tol_thresh_mult #tolerance threshold
    fl_tol_idxs = [i for i, j in enumerate(mm_list) if j >= tol_thresh]
    
    clear_borders_t, clear_borders_idxs, tmp_t, tmp_idxs = [], [], [], []        
    for i in range(0, int(np.floor(len(lfp)/win_len))):
        if i not in fl_tol_idxs:
            t = list(times[i * win_len : (i + 1) * win_len]) 
            tmp_t += [t[0], t[-1]]
            tmp_idxs += [i * win_len, (i + 1) * win_len]           
        if i in fl_tol_idxs or (i + 1) == int(np.floor(len(lfp)/win_len)):  
            if len(tmp_t) > 0 and (np.max(tmp_idxs) - np.min(tmp_idxs)) >= (min_len_mult * win_len):
                clear_borders_t.append([np.min(tmp_t), np.max(tmp_t)])
                clear_borders_idxs.append([np.min(tmp_idxs), np.max(tmp_idxs)])
            tmp_t, tmp_idxs = [], []
    
    if show == True:
        plt.plot(times, lfp, color = 'red')
        for chunk_idxs in clear_borders_idxs:
            plt.plot(np.arange(chunk_idxs[0], chunk_idxs[1]) / fs, 
                     list(lfp[chunk_idxs[0] : chunk_idxs[1]]), color = 'black')    
        plt.show()
    
    lfp_sample = []
    lfp_chunks = [list(lfp[ch_idxs[0] : ch_idxs[1]]) for ch_idxs in clear_borders_idxs]    
    for i in lfp_chunks: lfp_sample += i
    lfp_norm = (lfp - np.mean(lfp_sample)) / np.std(lfp_sample) #Нормализация!
    if len(clear_borders_idxs) > 1: #Зануляем артефактные отрезки сигнала
        cut_offs = [[clear_borders_idxs[i][1], clear_borders_idxs[i + 1][0]] for i in range(len(clear_borders_idxs) - 1)]
        for cut_idxs in cut_offs:
            lfp_norm[cut_idxs[0] : cut_idxs[1]] = 0    
    return lfp_norm, clear_borders_idxs #clear_borders_t, fl_tol_idxs,

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def fir_bandpass_filter(lfp_norm, fir_Order, band): #3000, theta_band [4, 11], delta_band [0.5, 3.5]
    fir = signal.firwin(fir_Order, band, fs = 1250, pass_zero = False) #band pass
    print("len(lfp_norm): ", len(lfp_norm))
    lfp_filtered = signal.filtfilt(fir, 1, lfp_norm)
    return lfp_filtered

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

#@jit(nopython=True)
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

    if end_idx[0] < start_idx[0]:
        start_idx.insert(0, 0)

    if start_idx[-1] > end_idx[-1]:
        end_idx.append(len(ripples_lfp) - 1)

    accept_intervals = (end_idx - start_idx) > accept_win * fs
    start_idx = start_idx[accept_intervals]
    end_idx = end_idx[accept_intervals]

    ripples_epoches = np.vstack([start_idx, end_idx])
    return ripples_epoches

@jit(nopython=True)
def get_theta_non_theta_epoches(theta_lfp, delta_lfp, lfp_norm, clear_borders_idxs, theta_thresh = 2, theta_min_dur_mult = 1, accept_win = 10, show = False):     
    #По алгоритму из (Kocsis B et al 2022)
    '''
    clear_borders_idxs - граничные индексы "чистых" участков lfp без артефактов
    theta_thresh - Пороговое значение отношения огибающих (тета к дельта) для выделения индексов участков с преобладанием тета-гармоник
    theta_min_dur_mult - множитель, задающий минимальную длительность тета-эпохи (умножается на число отсчетов в секунду)
    accept_win (лучше оставить равным 10) - для предварительного слияния очень близко расположенных тета-участков
    '''
    from copy import deepcopy as dc
    def merging_rel(intervals, rel_thresh_mult):
        #Слияние последующих интервалов с предыдущими при условии, что длительность разделяющих
        #их отрезков меньше суммы половин длительностей предшествующего и последующего интервалов
        #в rel_thresh_mult раз (субЪективный выбор)
        intervals_corr, incr_ep = [], dc(intervals[0]) #incr_ep - приращиваемый интервал
        for i in range(1, len(intervals)):
            sub_len = (intervals[i][1] - intervals[i][0]) / 2 + (intervals[i - 1][1] - intervals[i - 1][0])
            #sub_len = intervals[i][1] - intervals[i][0] #длительность данного тета-интрвала
            inter_len = intervals[i][0] - intervals[i - 1][1] #время между текущим и предыдущим тета-интервалом
            if sub_len >= inter_len * rel_thresh_mult:
                incr_ep[1] = intervals[i][1]
            else:
                intervals_corr.append(incr_ep)
                incr_ep = dc(intervals[i])
        return intervals_corr
    
    def merging_abs(intervals, abs_thresh):
        #Соединие интервалов при условии, что время между ними меньше выбранного порога        
        intervals_corr, incr_ep = [], dc(intervals[0])
        for i in range(1, len(intervals)):
            inter_len = intervals[i][0] - intervals[i - 1][1]
            if inter_len < abs_thresh:
                incr_ep[1] = intervals[i][1]
            else:
                intervals_corr.append(incr_ep)
                incr_ep = dc(intervals[i])
        return intervals_corr
    
    #Получение мгновенных амплитуд тета и дельта составляющих:                      
    sub_theta_env = np.abs(hilbert(theta_lfp))
    sub_delta_env = np.abs(hilbert(delta_lfp))    
    
    #Нахождение соотношения мгновенных амплитуд тета и дельта составляющих: 
    theta2delta = sub_theta_env / sub_delta_env
    theta2delta[theta2delta > 10] = 10 #Замена зашкаливающих значений 
    theta2delta = np.array([i if not np.isnan(i) else 0 for i in theta2delta])
              
    #Основной способ выделения первичных тета-эпох:
    theta_idxs = [i for i, j in enumerate(theta2delta) if j >= theta_thresh and lfp_norm[i] != 0]
    theta_epochs, borders = [], [theta_idxs[0]]
    for i in range(1, len(theta_idxs) - 1):
        if theta_idxs[i + 1] - theta_idxs[i] > accept_win:
            borders.append(theta_idxs[i])
            if len(borders) == 2:
                if borders[1] - borders[0] > 0:
                    theta_epochs.append(borders)
                borders = [theta_idxs[i + 1]]  
    '''
    #Дополнительный способ выделения границ тета-событий (для контроля, можно отключить):
    theta_idxs_control = [i if j >= theta_thresh and lfp_norm[i] != 0 else np.nan 
                          for i, j in enumerate(theta2delta)]
    theta_epochs_control, theta_subset = [], []
    for i in theta_idxs_control:
        if not np.isnan(i): 
            theta_subset.append(i)
        else:
            if len(theta_subset) > 1:
                theta_epochs_control.append([np.min(theta_subset), np.max(theta_subset)])
                theta_subset = []
    '''
    theta_epochs_corr = merging_rel(theta_epochs, 2)
    theta_epochs_corr = [i for i in theta_epochs_corr if i[1] - i[0] >= theta_min_dur_mult * 1250]
    theta_epochs_corr = merging_abs(theta_epochs_corr, 1250)
    #Получение границ дельта-эпох за счет инвертирования тета-интервалов: 
    delta_epochs = ([[0, theta_epochs_corr[0][0]]]
                    + [[theta_epochs_corr[i][1], theta_epochs_corr[i + 1][0]] for i in range(len(theta_epochs_corr) - 1)]) 
                    #+ [[theta_epochs_corr[-1][1], len(lfp_norm) - 1]])
    a = np.transpose(delta_epochs)[0] #Вырезание артефактных интервалов
    false_epochs_nums = [len(delta_epochs) - len(a[a >= i]) - 1 for i in np.transpose(clear_borders_idxs)[0]]
    delta_epochs_corr = [delta_epochs[i] for i in range(len(delta_epochs)) if i not in false_epochs_nums] 
    
    if show == True: #Визуальная проверка (задать свой адрес сохранения):              
        ix = 10000 #Для удобной развертки на одну картинку по 10000 отсчетов
        def get_idx_list(epoch_idxs):
            idxs = [np.nan for k in range(0, epoch_idxs[0][0])]
            for j in range(len(epoch_idxs)):
                idxs += [1 for k in range(epoch_idxs[j][0], epoch_idxs[j][1] + 1)]
                if j < len(epoch_idxs) - 1:
                    idxs += [np.nan for k in range(epoch_idxs[j][1] + 1, epoch_idxs[j + 1][0])]    
            return np.array(idxs)       
        #theta_uncorr_1 = [1 if not np.isnan(i) else i for i in theta_idxs_control] #исходные тета-эпохи
        theta_uncorr_2 = get_idx_list(theta_epochs) 
        #theta_control = get_idx_list(theta_epochs_control) * 2      
        theta_corr = get_idx_list(theta_epochs_corr) * 3
        delta_corr = get_idx_list(delta_epochs_corr) * 4
        for i in range(1, int(np.floor(1999359 / 10000))):
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (30, 10))                                        
            ax.plot(np.arange(len(lfp_norm))[(i - 1) * ix : i * ix] / 1250, theta_uncorr_2[(i - 1) * ix : i * ix], color = 'blue')             
            #ax.plot(np.arange(len(lfp_norm))[(i - 1) * ix : i * ix] / 1250, theta_control[(i - 1) * ix : i * ix], color = 'red')
            ax.plot(np.arange(len(theta_corr))[(i - 1) * ix : i * ix] / 1250, theta_corr[(i - 1) * ix : i * ix], color = 'magenta')        
            ax.plot(np.arange(len(delta_corr))[(i - 1) * ix : i * ix] / 1250, delta_corr[(i - 1) * ix : i * ix], color = 'green')
            ax.plot(np.arange(len(theta2delta))[(i - 1) * ix : i * ix] / 1250, theta2delta[(i - 1) * ix : i * ix], color = 'black')         
            fig.savefig("D:\\A_PopModels/test/" + 'sub_' + str(int((i - 1) * ix / 1250)) 
                        + '-' + str(int(i * ix / 1250)) + '_.png')
            
    return theta_epochs_corr, delta_epochs_corr


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




