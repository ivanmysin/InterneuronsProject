import os
import h5py
from params import rhythms_freqs_range, feasures_names
import pandas as pd
def main():
    sourses_path = '/media/ivan/Seagate Backup Plus Drive/Data/tranpsposed/'
    target_path = '/media/ivan/Seagate Backup Plus Drive/Data/table/feasures_table.hdf5'

    feasures_table = pd.DataFrame(columns=feasures_names)

    # Цикл по файлам директории preprocessing_1_results
    for filename in os.listdir(sourses_path):
        if filename[-5:] != '.hdf5': continue

        with h5py.File(sourses_path + filename, 'r') as sourse_hdf5:
            for electrode in sourse_hdf5.values():
                lfp_group = electrode['lfp']
                lfp_by_ranges = {}
                for rhythms_name in rhythms_freqs_range.keys():
                    lfp_by_ranges[rhythms_name] = lfp_group[rhythms_name][:]

                theta_epoches = lfp_group['theta_epoches'][:]
                non_theta_epoches = lfp_group['non_theta_epoches'][:]
                ripple_epoches = lfp_group['ripple_epoches'][:]
                spikes_group = electrode['spikes']
                for cluster_group in spikes_group.values():
                    spike_train = cluster_group['train'][:]

                    # Для каждого интернейрона находим его характеристики:
                    # Фазу тета-ритма и R
                    # Фазу медленного гамма-ритма и R в тета-состонии
                    # Фазу среднего гамма-ритма и R в тета-состонии
                    # Фазу быстрого гамма-ритма и R в тета-состонии
                    # Фазу медленного гамма-ритма и R в нетета-состонии
                    # Фазу среднего гамма-ритма и R в нетета-состонии
                    # Фазу быстрого гамма-ритма и R в нетета-состонии
                    # Фазу рипплов и R
                    # Частоты разрядов в тета-состониянии
                    # Частоты разрядов в нетета-состониянии

                    # заносим все в таблицу
                    #feasures_table.join()


        # Результаты сохраняем в виде таблицы объекты/признаки в файл директории results
        #  для классификации
        # сохраняем таблицу в файл feasures_table
        print(filename, " is processed")
        break

    feasures_table.to_hdf(target_path, mode='w')

if __name__ == "__main__":
    main()