import os
import mne
import pickle as pkl
from glob import glob
from mne_faster import (
    find_bad_epochs,
    find_bad_channels,
    find_bad_components,
    find_bad_channels_in_epochs,
)


def get_and_segregate_filepaths(path):
    all_file_paths = glob(path)

    rest_file_paths = [
        file_path
        for file_path in all_file_paths
        if os.path.basename(file_path)[-9] == "-"
    ]
    twob_file_paths = [
        file_path
        for file_path in all_file_paths
        if os.path.basename(file_path)[-9] == "2"
    ]

    return rest_file_paths, twob_file_paths


def read_filter(filepath):
    # reading data from set file
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)

    # dropping ecg1 and cz channels as cz is not present in all samples
    raw.drop_channels(
        ch_names=["ECG1", "Cz"],
        on_missing="ignore",  # can be set to raise, warn or ignore
    )

    # applying high pass filter with threshold 1hz
    high_pass_filtered_eeg = raw.filter(
        l_freq=1,
        h_freq=None,
        n_jobs=4,
        verbose=False,
    )

    # notch filtering with threshold 50hz
    filtered_eeg = high_pass_filtered_eeg.notch_filter(
        freqs=50,
        verbose=False,
    )

    return filtered_eeg


def eeg_event_extractor(filtered_eeg, task):
    # specifying the events to be included in annotations
    include_annots = [f"6{task}21", f"6{task}22"]

    # include only the specified events in the annotations
    include_mask = [
        annot in include_annots for annot in filtered_eeg.annotations.description
    ]

    included_annotations = mne.Annotations(
        onset=filtered_eeg.annotations.onset[include_mask],
        duration=filtered_eeg.annotations.duration[include_mask],
        description=filtered_eeg.annotations.description[include_mask],
    )
    filtered_eeg.set_annotations(included_annotations)

    # setting default eeg reference
    filtered_eeg, _ = mne.set_eeg_reference(
        filtered_eeg, ref_channels=[], copy=False, verbose=False
    )

    # extract events from the annotations and exclude any events not considered for epoching
    events, _ = mne.events_from_annotations(
        raw=filtered_eeg,
        event_id=None,
        verbose=False,
    )

    events = events[events[:, 2] <= len(include_annots)]
    event_id = {include_annots[i]: i + 1 for i in range(len(include_annots))}

    return events, event_id


def epoch_extractor(filtered_eeg, task):
    if task == "r":
        epochs = mne.make_fixed_length_epochs(
            raw=filtered_eeg,
            duration=1,
            preload=True,
            overlap=0.59,
            verbose=False,
        )

    else:
        events, event_id = eeg_event_extractor(filtered_eeg, task)

        # specify the time window for the epochs
        tmin, tmax = -0.05, 0.95

        epochs = mne.Epochs(
            raw=filtered_eeg,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
            preload=True,
            verbose=False,
        )

    return epochs


def mark_and_interpolate_bad_chnls(epochs):
    epochs.info["bads"] = find_bad_channels(
        epochs=epochs,
        eeg_ref_corr=False,
    )

    if len(epochs.info["bads"]) > 0:
        epochs.interpolate_bads()

    return epochs


def mark_and_drop_bad_epochs(epochs):
    bad_epochs = find_bad_epochs(
        epochs=epochs,
    )

    if len(bad_epochs) > 0:
        epochs.drop(bad_epochs)

    return epochs


def clean_bad_ica_components(epochs):
    ica = mne.preprocessing.ICA(0.99999999).fit(epochs)
    ica.exclude = find_bad_components(
        ica,
        epochs,
        use_metrics=["kurtosis", "power_gradient", "hurst", "median_gradient"],
    )

    ica.apply(epochs)

    # Need to re-baseline data after ICA transformation
    epochs.apply_baseline(epochs.baseline)

    return epochs


def interpolate_bad_channels_per_epoch(epochs):
    bad_channels_per_epoch = find_bad_channels_in_epochs(
        epochs=epochs,
        eeg_ref_corr=False,
    )

    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            ep = epochs[i]
            ep.info["bads"] = b
            ep.interpolate_bads()
            epochs._data[i, :, :] = ep._data[0, :, :]

    return epochs


def faster_artifact_correction(epochs):
    # Clean the data using FASTER

    # Step 1: mark and interpolate bad channels
    channel_interpolated_epochs = mark_and_interpolate_bad_chnls(epochs)

    # Step 2: mark and drop bad epochs
    good_epochs = mark_and_drop_bad_epochs(channel_interpolated_epochs)

    # Step 3: mark bad ICA components (using the build-in MNE functionality for this)
    cleaned_epochs = clean_bad_ica_components(good_epochs)

    # Step 4: mark bad channels for each epoch and interpolate them.
    interpolated_channels_per_epoch = interpolate_bad_channels_per_epoch(cleaned_epochs)

    return interpolated_channels_per_epoch


def eeg_prepocessor(path, task):
    eeg = read_filter(
        filepath=path,
    )

    epochs = epoch_extractor(
        filtered_eeg=eeg,
        task=task,
    )

    cleaned_epochs = faster_artifact_correction(
        epochs=epochs,
    )

    return cleaned_epochs


if __name__ == "__main__":
    rest_file_paths, twob_file_paths = get_and_segregate_filepaths(
        "../../BTP-files/final-dataset/*.set"
    )

    # list of length len(rest_files_path) with each object being an epoch.get_data().shape = [~144, 62, 500]
    rest_epochs = [eeg_prepocessor(rest_file, "r") for rest_file in rest_file_paths]

    with open("../preprocessed-data/rest_epochs.pkl", "wb") as f:
        pkl.dump(rest_epochs, f)

    # list of length len(two_files_path) with each object being an epoch.get_data().shape = [~144, 62, 501]
    twob_epochs = [eeg_prepocessor(twob_file, 2) for twob_file in twob_file_paths]

    with open("../preprocessed-data/twob_epochs.pkl", "wb") as f:
        pkl.dump(twob_epochs, f)
