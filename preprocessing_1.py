import os
import numpy as np
import h5py
from processing_lib import get_ripples_episodes_indexes, get_theta_non_theta_epoches, butter_bandpass_filter, clear_articacts
from params import rhythms_freqs_range, theta_epoches_params
from scipy.signal import hilbert

def select_files(sourses_path):
    SelectedFiles = []

    for path, _, files in os.walk(sourses_path):

        for file in files:
            if file.find(".hdf5") == -1:
                continue

            # print(file)
            sourse_hdf5 = h5py.File(path + '/' + file, "r")
            for ele_key, electrode in sourse_hdf5.items():
                try:
                    if electrode.attrs['brainZone'] != 'CA1':
                        continue
                except KeyError:
                    continue

                for cluster in electrode['spikes'].values():
                    try:
                        if cluster.attrs['type'] != 'Int' or cluster.attrs['quality'] != 'Nice':
                            continue
                    except KeyError:
                        continue

                    SelectedFiles.append(path + '/' + file)
                    break
                break
            sourse_hdf5.close()
    return SelectedFiles

def main():
    sourses_path = '/media/ivan/Seagate Backup Plus Drive/Data/myCRCNC/hc-3/'
    target_path = '/media/ivan/Seagate Backup Plus Drive/Data/tranpsposed/'

    SelectedFiles = select_files(sourses_path)

    for pathfile in SelectedFiles:
        sourse_hdf5 = h5py.File(pathfile, "r")
        file_name = pathfile.split("/")[-1]
        target_hdf5 = h5py.File(target_path + file_name, "w")

        for electrode_name, electrode in sourse_hdf5.items():
            try:
                if electrode.attrs['brainZone'] != 'CA1':
                    continue
            except KeyError:
                continue

            target_ele_group = target_hdf5.create_group(electrode_name)
            try:
                pyr_layer_number = electrode.attrs['pyramidal_layer']
            except KeyError:
                pyr_layer_number = 1

            channels_names = sorted( electrode['lfp'].keys() )
            lfp = electrode['lfp'][channels_names[pyr_layer_number - 1] ][:]
            lfp = lfp.astype(np.float64) / sourse_hdf5.attrs['amplification']
            lfp = clear_articacts(lfp)
            fs = electrode['lfp'].attrs['lfpSamplingRate']

            lfp_target_ele_group = target_ele_group.create_group('lfp')
            filtered_lfp = {}
            for rhythm_name, rhythm_range in rhythms_freqs_range.items():
                range_lfp = butter_bandpass_filter(lfp, rhythm_range[0], rhythm_range[1], fs, 3)
                range_lfp = hilbert(range_lfp)
                lfp_target_ele_group.create_dataset(rhythm_name, data=range_lfp)
                filtered_lfp[rhythm_name] = range_lfp

            theta_epoches, non_theta_epoches = get_theta_non_theta_epoches(filtered_lfp["theta"], filtered_lfp["delta"], fs,\
                                                        theta_epoches_params["theta_shreshold"], theta_epoches_params["accept_window_theta_shreshold"])
            lfp_target_ele_group.create_dataset('theta_epoches', data = theta_epoches)
            lfp_target_ele_group.create_dataset('non_theta_epoches', data = non_theta_epoches)

            ripple_epoches = get_ripples_episodes_indexes(filtered_lfp["ripples"], fs,  ripple_frqs = rhythms_freqs_range['ripples'])
            lfp_target_ele_group.create_dataset('ripple_epoches', data = ripple_epoches)


            spikes_target_ele_group = target_ele_group.create_group('spikes')
            for cluster_name, cluster in electrode['spikes'].items():
                try:
                    if cluster.attrs['type'] != 'Int' or cluster.attrs['quality'] != 'Nice':
                        continue
                except KeyError:
                    continue

                target_cluster = spikes_target_ele_group.create_group(cluster_name)

                target_cluster.create_dataset('train', data = cluster['train'][:]/sourse_hdf5.attrs['samplingRate'])

        print(file_name + " is processed")
        target_hdf5.close()
        sourse_hdf5.close()






if __name__ == "__main__":
    main()
