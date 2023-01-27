from collections import OrderedDict

import numpy as np
from skm_pyutils.table import list_to_df


def filter_good_units(unit_channels, sort_=True):
    # Very NB https://allensdk.readthedocs.io/en/latest/_static/examples/nb/visual_behavior_neuropixels_quality_metrics.html
    if sort_:
        unit_channels = unit_channels.sort_values(
            "probe_vertical_position", ascending=False
        )
    good_unit_filter = (
        (unit_channels["isi_violations"] < 0.4)  # Well isolated units
        & (unit_channels["nn_hit_rate"] > 0.9)  # Well isolated units
        & (
            unit_channels["amplitude_cutoff"] < 0.1
        )  # Units that have most of their activations
        & (unit_channels["presence_ratio"] > 0.9)  # Tracked for 90% of the recording
        # & (unit_channels["quality"] == "good") # Non-artifactual waveform
    )

    return unit_channels.loc[good_unit_filter]


def extract_useful_allen(recording, filter_units=False):
    session = recording.data
    if filter_units:
        # Removes noisy waveforms and units not in a brain area
        units = session.get_units(
            filter_by_validity=True,
            filter_out_of_brain_units=True,
        )
    else:
        units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on="peak_channel_id", right_index=True)
    unit_channels.to_csv("units.csv")

    # See https://allensdk.readthedocs.io/en/latest/_static/examples/nb/aligning_behavioral_data_to_task_events_with_the_stimulus_and_trials_tables.html
    stimulus_presentations = session.stimulus_presentations
    active_stimuli = stimulus_presentations[
        stimulus_presentations["active"] & stimulus_presentations["is_change"]
    ]
    passed = active_stimuli["rewarded"]
    trial_times = np.zeros(shape=(len(active_stimuli), 2))
    trial_times[:, 0] = active_stimuli["start_time"]
    trial_times[:, 1] = active_stimuli["end_time"]
    good_units = filter_good_units(unit_channels)
    good_units.to_csv("good_units.csv")

    good_spikes = {
        k: v for k, v in session.spike_times.items() if k in good_units.index
    }

    return trial_times, good_units, good_spikes, passed


def allen_to_general_bridge(recording):
    change_times, good_units, spike_times = extract_useful_allen(recording)


def one_to_general_bridge(recording):
    pass


def smooth_spike_train():
    pass


def filter_good_one_units(recording):
    # TODO also need to verify this filtering
    # TODO may be possible to compute our own to match allen
    results = {}
    for k, v in recording.data.items():
        if not k.startswith("probe"):
            continue
        unit_table = v[1]
        conditions = (
            (unit_table["presence_ratio"] > 0.9)
            & (unit_table["contamination"] < 0.4)
            & (unit_table["noise_cutoff"] < 25)
            & (unit_table["amp_median"] > 40 * 10**-6)
        )
        results[k] = unit_table.loc[conditions]
    return results


def create_spike_train_one(recording, good_unit_dict=None):
    results = {}
    for k, v in recording.data.items():
        if not k.startswith("probe"):
            continue

        spikes, clusters_df = v
        spike_train = OrderedDict()
        if good_unit_dict is None:
            index = clusters_df.index
        else:
            if isinstance(good_unit_dict[k], list):
                index = good_unit_dict[k]
            else:
                index = good_unit_dict[k].index
        for val in index:
            spike_train[val] = []
        for i in range(len(spikes["depths"])):
            cluster = spikes["clusters"][i]
            if cluster in spike_train:
                spike_train[cluster].append(spikes["times"][i])

        for k2, v2 in spike_train.items():
            spike_train[k2] = np.array(v2).reshape((1, -1))

        results[k] = spike_train

    return results


def simple_behaviour_compare(spike_train, passes, trial_times):
    """
    Compare matrix of spike times in a window around trials to compare pass and fail.

    """
    result_list = []

    for p, t in zip(passes, trial_times):
        start_time, end_time = t
        for k, v in spike_train.items():
            part_of_interest = np.nonzero(
                np.logical_and((v >= start_time), (v < end_time))
            )[0]
            piece = v[part_of_interest]
            firing_rate_in_window = len(piece) / (end_time - start_time)
            result_list.append([p, k, firing_rate_in_window])

    result_df = list_to_df(result_list, ["Passed", "Unit", "Num spikes"])

    return result_df.groupby(["Passed", "Unit"]).mean()
