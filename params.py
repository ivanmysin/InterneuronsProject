
rhythms_freqs_range = {
    # Гц
    "delta": [1, 4],
    "theta": [4, 12],
    "slow_gamma": [25, 45],
    "middle_gamma": [45, 80],
    "fast_gamma": [80, 150],
    "ripples": [120, 250],
}

theta_epoches_params = {
    "theta_threshold" : 2,
    "accept_window_theta_shreshold" : 2, # секунды
}

circ_means_options = {
    "delta": "uniform",
    "theta": "uniform",
    "slow_gamma": 'normalized',
    "middle_gamma": 'normalized',
    "fast_gamma": 'normalized',
    "ripples": 'normalized',

}

ripples_detec = {
    "threshold" : 3,
    "accept_win" : 0.02
}


feasures_names = ["theta_phi", "theta_R"]
# gamma phases in theta state, s - slow, m - medium, f - fast
feasures_names.extend(["ts_gamma_s_phi", "ts_gamma_s_R"])
feasures_names.extend(["ts_gamma_m_phi", "ts_gamma_m_R"])
feasures_names.extend(["ts_gamma_f_phi", "ts_gamma_f_R"])

# gamma phases in non-theta state, s - slow, m - medium, f - fast
feasures_names.extend(["non_ts_gamma_s_phi", "non_ts_gamma_s_R"])
feasures_names.extend(["non_ts_gamma_m_phi", "non_ts_gamma_m_R"])
feasures_names.extend(["non_ts_gamma_f_phi", "non_ts_gamma_f_R"])

# phases during ripples
feasures_names.extend(["ripples_phi", "ripples_R"])

# spike rate in theta, non-theta state and ripples
feasures_names.extend(["ts_spike_rate", "ts_spike_rate_std"])
feasures_names.extend(["non_ts_spike_rate", "non_ts_std_spike_rate"])
feasures_names.extend(["ripples_spike_rate", "ripples_spike_rate_std"])
