import numpy as np
import os
import h5py
from params import rhythms_freqs_range, feasures_names, circ_means_options
import pandas as pd
import processing_lib as plib
import multiprocessing
def run_processing_2(params):
    sourses_files = params[0]
    sourses_path = params[1]
    samplingRate = params[2]

    feasures_table = pd.DataFrame(columns=feasures_names)
    # Цикл по файлам директории preprocessing_1_results
    for filename in sourses_files:
        if filename[-5:] != '.hdf5': continue

        with h5py.File(sourses_path + filename, 'r') as sourse_hdf5:
            for electrode in sourse_hdf5.values():
                lfp_group = electrode['lfp']
                lfp_by_ranges = {}
                for rhythms_name in rhythms_freqs_range.keys():
                    lfp_by_ranges[rhythms_name] = lfp_group[rhythms_name][:]

                theta_epoches = lfp_group['theta_epoches'][:]
                non_theta_epoches = lfp_group['non_theta_epoches'][:]
                ripple_epoches = lfp_group['ripple_epoches'][:].T

                spikes_group = electrode['spikes']
                for cluster_group in spikes_group.values():
                    neuron_feasures = {}

                    spike_train = cluster_group['train'][:]
                    if spike_train.size == 0: continue

                    spike_train = np.round(spike_train * samplingRate).astype(np.int64)

                    # Находим среднюю частоту разрядов в тета эпохах
                    ts_spike_rate, ts_spike_rate_std = plib.get_mean_spike_rate_by_epoches(theta_epoches, spike_train, samplingRate)
                    non_ts_spike_rate, non_ts_std_spike_rate = plib.get_mean_spike_rate_by_epoches(non_theta_epoches, spike_train, samplingRate)
                    ripples_spike_rate, ripples_spike_rate_std = plib.get_mean_spike_rate_by_epoches(ripple_epoches, spike_train, samplingRate)

                    neuron_feasures["ts_spike_rate"] = ts_spike_rate
                    neuron_feasures["ts_spike_rate_std"] = ts_spike_rate_std
                    neuron_feasures["non_ts_spike_rate"] = non_ts_spike_rate
                    neuron_feasures["non_ts_std_spike_rate"] = non_ts_std_spike_rate
                    neuron_feasures["ripples_spike_rate"] = ripples_spike_rate
                    neuron_feasures["ripples_spike_rate_std"] = ripples_spike_rate_std

                    spikes_during_theta_epoches = plib.get_over_all_epoches(theta_epoches, spike_train)
                    spikes_during_non_theta_epoches = plib.get_over_all_epoches(non_theta_epoches, spike_train)
                    spikes_during_ripples_epoches = plib.get_over_all_epoches(ripple_epoches, spike_train)

                    theta_phase, theta_R = plib.get_circular_mean_R(lfp_by_ranges["theta"], spikes_during_theta_epoches, mean_calculation=circ_means_options["theta"])
                    ripples_phase, ripples_R = plib.get_circular_mean_R(lfp_by_ranges["ripples"], spikes_during_ripples_epoches, mean_calculation=circ_means_options["ripples"])

                    neuron_feasures["theta_phi"] = theta_phase
                    neuron_feasures["theta_R"] = theta_R
                    neuron_feasures["ripples_phi"] = ripples_phase
                    neuron_feasures["ripples_R"] = ripples_R

                    for rhythm_name, lfp_range in lfp_by_ranges.items():
                        if not("gamma" in rhythm_name): continue
                        gamma_phase, gamma_R = plib.get_circular_mean_R(lfp_range, spikes_during_theta_epoches, mean_calculation=circ_means_options[rhythm_name])

                        neuron_feasures["ts_gamma_" + rhythm_name[0] + "_phi"] = gamma_phase
                        neuron_feasures["ts_gamma_" + rhythm_name[0] + "_R"] = gamma_R

                    for rhythm_name, lfp_range in lfp_by_ranges.items():
                        if not("gamma" in rhythm_name): continue
                        gamma_phase, gamma_R = plib.get_circular_mean_R(lfp_range, spikes_during_non_theta_epoches, mean_calculation=circ_means_options[rhythm_name])
                        neuron_feasures["non_ts_gamma_" + rhythm_name[0] + "_phi"] = gamma_phase
                        neuron_feasures["non_ts_gamma_" + rhythm_name[0] + "_R"] = gamma_R

                    # заносим все в таблицу
                    feasures_table.loc[len(feasures_table)] = pd.Series(neuron_feasures)
        print(filename, " is processed")

def main():
    samplingRate = 1250 # !!!!!!!!!
    sourses_path = '/media/ivan/Seagate Backup Plus Drive/Data/tranpsposed/'
    #'/media/usb/Data/InterneuronsProject/preprocessing_1/'
    target_path = './results/feasures_table.hdf5'

    sourses_files = os.listdir(sourses_path)
    n_cpu = multiprocessing.cpu_count()
    FileByThreds = []
    for idx in range(n_cpu):
        params = [sourses_files[idx::n_cpu], sourses_path, samplingRate]
        FileByThreds.append(params)

    with multiprocessing.Pool(n_cpu) as run_pool:
        feasures_tables = run_pool.map(run_processing_2, FileByThreds)
    feasures_table = pd.concat(feasures_tables)



    print(feasures_table)
    # Результаты сохраняем в виде таблицы объекты/признаки в файл директории results для классификации
    # сохраняем таблицу в файл feasures_table
    feasures_table.to_hdf(target_path, key='interneurons_feasures', mode='w')

if __name__ == "__main__":
    main()