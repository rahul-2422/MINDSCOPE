import pickle as pkl

import numpy as np
import pandas as pd
from preprocessing import read_filter
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_classif


def epoch_data_extractor(rest_epochs, twob_epochs):
    rest_epochs_data = [
        rest_epochs[i].get_data(verbose=False) for i in range(len(rest_epochs))
    ]
    rest_epochs_arr = np.vstack(rest_epochs_data)

    twob_epochs_data = [
        twob_epochs[i].get_data(verbose=False) for i in range(len(twob_epochs))
    ]
    twob_epochs_arr = np.vstack(twob_epochs_data)
    twob_epochs_arr = twob_epochs_arr[:, :, :500]

    return rest_epochs_arr, twob_epochs_arr


def mean_psd(channel_array, Fs=500):
    n_channels, n_epochs, _ = channel_array.shape
    BandF = [12, 30, 100]
    PMean = np.zeros([n_channels, n_epochs, len(BandF) - 1])

    for ch in range(n_channels):
        for epoch in range(n_epochs):
            f, Psd = welch(channel_array[ch, epoch, :], Fs)

            if np.any(np.isnan(Psd)):
                non_nan = Psd[~np.isnan(Psd)]
                nan_avg = np.mean(non_nan)
                Psd[np.isnan(Psd)] = nan_avg

            for k in range(len(BandF) - 1):
                fr = np.where((f > BandF[k]) & (f <= BandF[k + 1]))
                PMean[ch, epoch, k] = np.mean(Psd[fr])

    return PMean


def best_channel_selector(channel_features, label_array):
    mutual_info = []

    for channel in range(channel_features.shape[0]):
        mutual_info_ch = mutual_info_classif(
            channel_features[channel, :, :], label_array
        )
        mutual_info.append(np.sum(mutual_info_ch))

    mutual_info = pd.Series(data=mutual_info, index=channels)
    mutual_info = mutual_info.sort_values(ascending=False)
    return mutual_info.index[0]


def get_channel_features(channel_set, all_channel_features, channels):
    ch_indices = [channels.index(ch_name) for ch_name in channel_set]

    ch_features = all_channel_features[ch_indices, :, :]
    channel_features = np.empty((ch_features.shape[1], 0))

    for ch in range(ch_features.shape[0]):
        ch_feature = ch_features[ch].reshape(ch_features.shape[1], -1)
        channel_features = np.hstack((channel_features, ch_feature))

    return channel_features


def joint_mutual_info(
    channels,
    channel_set,
    lable_array,
    all_channel_features,
    candidate_channel_set,
    ranked_channel_features,
):
    joint_candidate_mutual_info = []

    for channel in channel_set:
        candidate_channel_set.append(channel)

        ch_index = channels.index(candidate_channel_set[-1])
        ch_i_features = all_channel_features[ch_index, :, :]
        candidate_channel_features = np.hstack((ranked_channel_features, ch_i_features))

        mutual_info_ch_i = mutual_info_classif(candidate_channel_features, label_array)
        joint_candidate_mutual_info.append(np.sum(mutual_info_ch_i))

    joint_candidate_mutual_info = pd.Series(
        data=joint_candidate_mutual_info, index=channel_set
    )

    joint_candidate_mutual_info = joint_candidate_mutual_info.sort_values(
        ascending=False
    )

    return joint_candidate_mutual_info.index[0]


def rank_channel_set(
    channels, channel_set, label_array, best_channel, all_channel_features
):
    channel_set.remove(best_channel)
    ranked_channel_set = [best_channel]

    while len(channel_set) > 0:
        candidate_channel_set = []
        ranked_channel_features = get_channel_features(
            ranked_channel_set, all_channel_features, channels
        )
        candidate_channel_set.append(ranked_channel_set)
        best_joint_channel = joint_mutual_info(
            channels,
            channel_set,
            label_array,
            all_channel_features,
            candidate_channel_set,
            ranked_channel_features,
        )
        ranked_channel_set.append(best_joint_channel)
        channel_set.remove(best_joint_channel)
        candidate_channel_set.clear()

    return ranked_channel_set


if __name__ == "__main__":
    with open("../preprocessed-data/rest_epochs.pkl", "rb") as f:
        rest_epochs = pkl.load(f)

    with open("../preprocessed-data/twob_epochs.pkl", "rb") as f:
        twob_epochs = pkl.load(f)

    rest_data, twob_data = epoch_data_extractor(rest_epochs, twob_epochs)

    del rest_epochs, twob_epochs

    data_array = np.vstack((rest_data, twob_data))

    rest_labels = [0] * len(rest_data)
    twob_labels = [1] * len(twob_data)
    label_array = np.hstack((rest_labels, twob_labels))

    del rest_data, rest_labels, twob_data, twob_labels

    channel_array = np.transpose(data_array, (1, 0, 2))
    all_channel_features = mean_psd(channel_array)

    eeg = read_filter("../../BTP-files/final-dataset/sub-24-s1-rest.set")
    channels = eeg.ch_names
    channel_set = channels.copy()
    best_channel = best_channel_selector(all_channel_features, label_array)

    ranked_channel_set = rank_channel_set(
        channels, channel_set, label_array, best_channel, all_channel_features
    )

    del label_array

    with open("../channel-selection-data/ranked_channels.pkl", "wb") as f:
        pkl.dump(ranked_channel_set, f)

    ranked_channels = ranked_channel_set[:14]

    del ranked_channel_set

    with open("../preprocessed-data/rest_epochs.pkl", "rb") as f:
        rest_epochs = pkl.load(f)

    with open("../preprocessed-data/twob_epochs.pkl", "rb") as f:
        twob_epochs = pkl.load(f)

    rest_epochs_2 = [
        rest_epochs[i].pick_channels(ranked_channels) for i in range(len(rest_epochs))
    ]
    twob_epochs_2 = [
        twob_epochs[i].pick_channels(ranked_channels) for i in range(len(twob_epochs))
    ]

    rest_data_2, twob_data_2 = epoch_data_extractor(rest_epochs_2, twob_epochs_2)

    rest_labels = [0] * len(rest_data_2)
    twob_labels = [1] * len(twob_data_2)
    label_array = np.hstack((rest_labels, twob_labels))

    with open("../channel-selection-data/chnl_selected_labels.pkl", "wb") as f:
        pkl.dump(label_array, f)

    del rest_labels, twob_labels, label_array

    channel_selected_data = np.vstack((rest_data_2, twob_data_2))

    with open("../channel-selection-data/chnl_selected_data.pkl", "wb") as f:
        pkl.dump(channel_selected_data, f)

    del rest_data_2, twob_data_2, channel_selected_data
