import pickle as pkl

import numpy as np
import pywt
from scipy import signal


def flatten(data):
    flattened_data = data.reshape(data.shape[0], -1)
    return flattened_data


def hjorth_mean(eeg_data):
    """
    Calculate the Hjorth parameters of EEG data and return the mean values across channels for each epoch.
    Args:
        eeg_data (ndarray): EEG data of shape (number_of_epochs, number_of_channels, number_of_datapoints_per_epoch).
    Returns:
        mean_activity (ndarray): Mean activity parameter of shape (number_of_epochs,).
        mean_mobility (ndarray): Mean mobility parameter of shape (number_of_epochs,).
        mean_complexity (ndarray): Mean complexity parameter of shape (number_of_epochs,).
    """
    n_epochs, n_channels, n_datapoints = eeg_data.shape
    mean_activity = np.zeros((n_epochs,))
    mean_mobility = np.zeros((n_epochs,))
    mean_complexity = np.zeros((n_epochs,))
    for i in range(n_epochs):
        activity = 0
        mobility = 0
        complexity = 0
        for j in range(n_channels):
            signal = eeg_data[i, j, :]
            diff1 = np.diff(signal)
            diff2 = np.diff(signal, n=2)
            var_zero = np.var(signal)
            var_d1 = np.var(diff1)
            var_d2 = np.var(diff2)
            activity += var_zero
            mobility += np.sqrt(var_d1 / var_zero)
            complexity += np.sqrt(var_d2 / var_d1) / np.sqrt(var_d1 / var_zero)
        mean_activity[i] = activity / n_channels
        mean_mobility[i] = mobility / n_channels
        mean_complexity[i] = complexity / n_channels
    return mean_activity, mean_mobility, mean_complexity


def kurtosis_feature(eeg_data):
    num_epochs, num_channels, num_datapoints_per_epoch = eeg_data.shape
    result = np.zeros(num_epochs)
    for i in range(num_epochs):
        epoch_data = eeg_data[i, :, :]
        epoch_mean = np.mean(epoch_data, axis=1)
        epoch_std = np.std(epoch_data, axis=1, ddof=1)
        epoch_kurtosis = (
            np.mean((epoch_data.T - epoch_mean) ** 4, axis=0) / epoch_std**4 - 3
        )
        result[i] = np.mean(epoch_kurtosis)
    return result


def wavelet_features(epoch):
    num_epochs, num_channels, num_samples = epoch.shape
    cA_mean = np.zeros((num_epochs, num_channels))
    cA_std = np.zeros((num_epochs, num_channels))
    cA_Energy = np.zeros((num_epochs, num_channels))
    cD_mean = np.zeros((num_epochs, num_channels))
    cD_std = np.zeros((num_epochs, num_channels))
    cD_Energy = np.zeros((num_epochs, num_channels))
    Entropy_D = np.zeros((num_epochs, num_channels))
    Entropy_A = np.zeros((num_epochs, num_channels))
    wfeatures = np.zeros((num_epochs, 7 * num_channels))

    for i in range(num_epochs):
        for j in range(num_channels):
            cA, cD = pywt.dwt(epoch[i, j, :], "coif1")
            cA_mean[i, j] = np.mean(cA)
            cA_std[i, j] = np.abs(np.std(cA))
            cA_Energy[i, j] = np.sum(np.square(cA))
            cD_mean[i, j] = np.mean(cD)
            cD_std[i, j] = np.abs(np.std(cD))
            cD_Energy[i, j] = np.sum(np.square(cD))
            Entropy_D[i, j] = np.sum(np.square(cD) * np.log(np.square(cD)))
            Entropy_A[i, j] = np.sum(np.square(cA) * np.log(np.square(cA)))

    wfeatures[:, 0::7] = cA_mean
    wfeatures[:, 1::7] = cA_std
    wfeatures[:, 2::7] = cA_Energy
    wfeatures[:, 3::7] = cD_mean
    wfeatures[:, 4::7] = cD_std
    wfeatures[:, 5::7] = cD_Energy
    wfeatures[:, 6::7] = Entropy_D + Entropy_A

    return wfeatures


def maxPwelch_epochs(epochs, Fs):
    n_epochs, n_channels, n_samples_per_epoch = epochs.shape
    BandF = [12, 30, 100]
    PMax = np.zeros([n_epochs, n_channels, len(BandF) - 1])

    for i in range(n_epochs):
        for j in range(n_channels):
            f, Psd = signal.welch(epochs[i, j, :], Fs)

            if np.any(np.isnan(Psd)):
                nonnan_values = Psd[~np.isnan(Psd)]
                nan_average = np.mean(nonnan_values)
                Psd[np.isnan(Psd)] = nan_average

            for k in range(len(BandF) - 1):
                fr = np.where((f > BandF[k]) & (f <= BandF[k + 1]))
                PMax[i, j, k] = np.max(Psd[fr])

    return PMax


if __name__ == "__main__":
    with open("../channel-selection-data/chnl_selected_data.pkl", "rb") as f:
        epoch_data = pkl.load(f)

    hjorth = hjorth_mean(epoch_data)
    hjorth_list = np.concatenate(
        (hjorth[0][:, np.newaxis], hjorth[1][:, np.newaxis], hjorth[2][:, np.newaxis]),
        axis=1,
    )
    del hjorth

    kurtosis = kurtosis_feature(epoch_data)

    wavelet = wavelet_features(epoch_data)

    psd = maxPwelch_epochs(epoch_data, 500)
    psd_2d = flatten(psd)
    del psd

    feature_vector = np.concatenate(
        (hjorth_list, kurtosis[:, np.newaxis], wavelet, psd_2d),
        axis=1,
    )

    with open("../features-data/features_data.pkl", "wb") as f:
        pkl.dump(feature_vector, f)
