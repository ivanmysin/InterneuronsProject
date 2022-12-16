import pandas as pd
import numpy as np
import h5py
from processing_lib import get_ripples_episodes_indexes, get_theta_non_theta_epoches, butter_bandpass_filter
from params import rhythms_freqs_range


def find_cells(path, zone, type = 'i'):

    df = pd.read_csv(path, index_col = 0, header = None)
    cells = df[(df[5] == zone) & (df[14] == type)]
    return cells

def find_sessions(path, cells):

    df = pd.read_csv(path, index_col=0, header=None)[[1, 2]]
    x = df[1].isin(cells[1])
    df = df[x]
    return df

def main():

    cells = find_cells('hc3-cell.csv', 'CA1')
    sessions = find_sessions('hc3-session.csv', cells)
    files = sessions[2]
    fs = 10000
    for i in files:

            lfp = np.fromfile(i + '.eeg', dtype = np.short)[:5000]

            theta_frqs = rhythms_freqs_range['theta']
            delta_frqs = rhythms_freqs_range['delta']
            slow_gamma_frqs = rhythms_freqs_range['slow_gamma']
            middle_gamma_frqs = rhythms_freqs_range['middle_gamma']
            fast_gamma_frqs = rhythms_freqs_range['fast_gamma']
            ripples_frqs = rhythms_freqs_range['ripples']

            theta_lfp = butter_bandpass_filter(lfp, theta_frqs[0], theta_frqs[1], fs, 4)
            delta_lfp = butter_bandpass_filter(lfp, delta_frqs[0], delta_frqs[1], fs, 4)
            slow_gamma_lfp = butter_bandpass_filter(lfp, slow_gamma_frqs[0], slow_gamma_frqs[1], fs, 4)
            middle_gamma_lfp = butter_bandpass_filter(lfp, middle_gamma_frqs[0], middle_gamma_frqs[1], fs, 4)
            fast_gamma_lfp = butter_bandpass_filter(lfp, fast_gamma_frqs[0], fast_gamma_frqs[1], fs, 4)
            ripples_lfp = butter_bandpass_filter(lfp, ripples_frqs[0], ripples_frqs[1], fs, 4)

            theta_epoches, non_theta_epoches = get_theta_non_theta_epoches(theta_lfp, delta_lfp)
            ripple_epoches = get_ripples_episodes_indexes(lfp, fs)

            hf = h5py.File('results/' + i, 'w')
            g1 = hf.create_group('filtered frequencies')
            g2 = hf.create_group('epoches')
            g1.create_dataset('theta_lfp', data = theta_lfp)
            g1.create_dataset('delta_lfp', data = delta_lfp)
            g1.create_dataset('slow_gamma_lfp', data = slow_gamma_lfp)
            g1.create_dataset('middle_gamma_lfp', data = middle_gamma_lfp)
            g1.create_dataset('fast_gamma_lfp', data = fast_gamma_lfp)
            g1.create_dataset('ripples_lfp', data = ripples_lfp)

            g2.create_dataset('theta_epoches', data = theta_epoches)
            g2.create_dataset('non_theta_epoches', data = non_theta_epoches)
            g2.create_dataset('ripple_epoches', data = ripple_epoches)


if __name__ == "__main__":
    main()
